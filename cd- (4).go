// loan_eligibility_engine.go

package eligibility

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/lendingclub/platform/bureau"
	"github.com/lendingclub/platform/eventbus"
	"github.com/lendingclub/platform/identity"
	"github.com/lendingclub/platform/lending/model"
	"github.com/lendingclub/platform/scoring"
	"github.com/lendingclub/platform/telemetry"
	"go.uber.org/zap"
)

// ---------------------------------------------------------------------------
// Policy constants  (source: Credit Policy CP-PL-2023-07)
// ---------------------------------------------------------------------------

const (
	minFICOScore           = 580
	maxDTI                 = 0.65
	minLoanAmountCents     = 100000      // $1,000
	maxLoanAmountCents     = 4000000     // $40,000
	minCreditHistoryMonths = 12
	maxConcurrentLCLoans   = 2
	maxInquiries6Months    = 6
	bankruptcyLookbackYrs  = 7
	seriousDelinqDPD       = 90         // DPD threshold for "serious delinquency"
	seriousDelinqLookback  = 365        // days

	// Offer pricing: prime rate + risk margin (bps)
	primeRateBPS = 850 // 8.50% as of policy effective date
)

// riskMarginBPS defines the spread over prime rate for each risk band.
// Source: Pricing Policy PP-PL-2024-01.
var riskMarginBPS = map[string]int{
	"A1": 120, "A2": 175, "A3": 230,
	"B1": 295, "B2": 360, "B3": 430,
	"C1": 510, "C2": 595, "C3": 685,
	"D":  790, "E":  920, "F":  1080,
	"G":  1290,
}

// autoDenyBands are risk bands that result in automatic denial.
var autoDenyBands = map[string]bool{"G": true}

// manualReviewBands require additional underwriter review before approval.
var manualReviewBands = map[string]bool{"F": true, "E": true}

// eligibleTerms defines the set of allowed loan terms in months.
var eligibleTerms = map[int]bool{36: true, 60: true}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// DeclineCode is a machine-readable identifier for a decline reason.
type DeclineCode string

const (
	DeclineFICOBelowMin             DeclineCode = "FICO_BELOW_MINIMUM"
	DeclineDTIExceedsHardCutoff     DeclineCode = "DTI_EXCEEDS_HARD_CUTOFF"
	DeclineActiveBankruptcy         DeclineCode = "ACTIVE_BANKRUPTCY"
	DeclineSeriousDelinquency       DeclineCode = "RECENT_SERIOUS_DELINQUENCY"
	DeclineExcessiveInquiries       DeclineCode = "EXCESSIVE_RECENT_INQUIRIES"
	DeclineInsufficientHistory      DeclineCode = "INSUFFICIENT_CREDIT_HISTORY"
	DeclineMaxLCLoansExceeded       DeclineCode = "MAX_CONCURRENT_LC_LOANS"
	DeclineAmountBelowMin           DeclineCode = "AMOUNT_BELOW_MINIMUM"
	DeclineAmountAboveMax           DeclineCode = "AMOUNT_ABOVE_MAXIMUM"
	DeclineInvalidTerm              DeclineCode = "INVALID_LOAN_TERM"
	DeclineIdentityFailed           DeclineCode = "IDENTITY_VERIFICATION_FAILED"
	DeclineStateIneligible          DeclineCode = "STATE_NOT_ELIGIBLE"
	DeclineRiskBandAutoDeny         DeclineCode = "RISK_BAND_AUTO_DENY"
	DeclineFraudFlag                DeclineCode = "FRAUD_OR_OFAC_FLAG"
	DeclineInvalidIncome            DeclineCode = "INVALID_INCOME"
)

// Outcome represents the underwriting decision outcome.
type Outcome string

const (
	OutcomeApproved     Outcome = "APPROVED"
	OutcomeDeclined     Outcome = "DECLINED"
	OutcomeManualReview Outcome = "MANUAL_REVIEW"
	OutcomePending      Outcome = "PENDING"
)

// Decision is the fully resolved underwriting decision for an application.
type Decision struct {
	ApplicationID  string        `json:"application_id"`
	MemberID       string        `json:"member_id"`
	Outcome        Outcome       `json:"outcome"`
	DeclineCodes   []DeclineCode `json:"decline_codes,omitempty"`
	DeclineReasons []string      `json:"decline_reasons,omitempty"`
	RiskBand       string        `json:"risk_band,omitempty"`
	FICOScore      int           `json:"fico_score,omitempty"`
	PDEstimate     float64       `json:"pd_estimate,omitempty"`
	Offer          *LoanOffer    `json:"offer,omitempty"`
	DecidedAt      time.Time     `json:"decided_at"`
	ModelVersion   string        `json:"model_version,omitempty"`
}

// LoanOffer contains the terms offered to an approved applicant.
type LoanOffer struct {
	OfferedAmountCents   int64   `json:"offered_amount_cents"`
	TermMonths           int     `json:"term_months"`
	AnnualRateBPS        int     `json:"annual_rate_bps"`   // total APR in basis points
	MonthlyPaymentCents  int64   `json:"monthly_payment_cents"`
	OriginationFeeBPS    int     `json:"origination_fee_bps"`
	OriginationFeeCents  int64   `json:"origination_fee_cents"`
	TotalInterestCents   int64   `json:"total_interest_cents"`
	EffectiveAPRBPS      int     `json:"effective_apr_bps"` // includes origination fee
	OfferExpiresAt       time.Time `json:"offer_expires_at"`
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

// Engine is the primary loan eligibility decision engine.
type Engine struct {
	bureauClient    bureau.Client
	scoringClient   scoring.Client
	identityClient  identity.Client
	loanRepository  LoanRepository
	eventPublisher  eventbus.Publisher
	metrics         telemetry.Metrics
	logger          *zap.Logger
	primeRateBPS    int           // injected to allow test overrides
	offerTTL        time.Duration // how long an offer remains valid
}

// NewEngine constructs an Engine with required dependencies.
func NewEngine(
	bureauClient bureau.Client,
	scoringClient scoring.Client,
	identityClient identity.Client,
	loanRepo LoanRepository,
	publisher eventbus.Publisher,
	metrics telemetry.Metrics,
	logger *zap.Logger,
) *Engine {
	return &Engine{
		bureauClient:   bureauClient,
		scoringClient:  scoringClient,
		identityClient: identityClient,
		loanRepository: loanRepo,
		eventPublisher: publisher,
		metrics:        metrics,
		logger:         logger,
		primeRateBPS:   primeRateBPS,
		offerTTL:       30 * 24 * time.Hour, // offer valid for 30 days
	}
}

// Evaluate orchestrates the full eligibility determination for a loan application.
// It runs identity, bureau, scoring, and policy evaluation in sequence
// (parallelizing where dependencies permit) and returns a Decision.
func (e *Engine) Evaluate(ctx context.Context, app *model.LoanApplication) (*Decision, error) {
	start := time.Now()
	log := e.logger.With(zap.String("application_id", app.ID), zap.String("member_id", app.MemberID))
	log.Info("starting eligibility evaluation")

	defer func() {
		e.metrics.RecordHistogram("lee.evaluate.duration_ms", float64(time.Since(start).Milliseconds()))
	}()

	// ── Stage 1: Request-level validation ──────────────────────────────────
	if codes, reasons := e.validateRequest(app); len(codes) > 0 {
		return e.decline(app.ID, app.MemberID, codes, reasons, "", 0, 0), nil
	}

	// ── Stage 2: Identity verification (hard stop) ─────────────────────────
	idResult, err := e.identityClient.Verify(ctx, identity.VerifyRequest{
		SSN:         app.Applicant.SSN,
		FirstName:   app.Applicant.FirstName,
		LastName:    app.Applicant.LastName,
		DateOfBirth: app.Applicant.DateOfBirth,
		AddressLine1: app.Applicant.AddressLine1,
		ZipCode:     app.Applicant.ZipCode,
	})
	if err != nil {
		return nil, fmt.Errorf("identity verification unavailable: %w", err)
	}
	if !idResult.Passed {
		e.metrics.Increment("lee.decline.identity_failed")
		return e.decline(app.ID, app.MemberID,
			[]DeclineCode{DeclineIdentityFailed},
			[]string{"Identity verification did not pass"},
			"", 0, 0), nil
	}
	if idResult.OFACFlagged || idResult.FraudFlagged {
		e.metrics.Increment("lee.decline.fraud_flag")
		return e.decline(app.ID, app.MemberID,
			[]DeclineCode{DeclineFraudFlag},
			[]string{"Application flagged during pre-screen checks"},
			"", 0, 0), nil
	}

	// ── Stage 3: Bureau pull (parallel with active loan count fetch) ────────
	type bureauResult struct {
		report *bureau.Report
		err    error
	}
	type loanCountResult struct {
		count int
		err   error
	}

	bureauCh := make(chan bureauResult, 1)
	loanCountCh := make(chan loanCountResult, 1)

	go func() {
		r, err := e.bureauClient.Pull(ctx, bureau.PullRequest{
			SSN:         app.Applicant.SSN,
			DateOfBirth: app.Applicant.DateOfBirth,
			FirstName:   app.Applicant.FirstName,
			LastName:    app.Applicant.LastName,
		})
		bureauCh <- bureauResult{r, err}
	}()

	go func() {
		c, err := e.loanRepository.CountActiveLoans(ctx, app.Applicant.SSN)
		loanCountCh <- loanCountResult{c, err}
	}()

	br := <-bureauCh
	if br.err != nil {
		return nil, fmt.Errorf("bureau pull failed: %w", br.err)
	}
	report := br.report

	lcr := <-loanCountCh
	if lcr.err != nil {
		return nil, fmt.Errorf("active loan count unavailable: %w", lcr.err)
	}
	activeLCLoans := lcr.count

	// ── Stage 4: Policy pre-screen ──────────────────────────────────────────
	codes, reasons := e.evaluatePolicyPreScreen(app, report, activeLCLoans)
	if len(codes) > 0 {
		for _, c := range codes {
			e.metrics.Increment("lee.decline." + string(c))
		}
		return e.decline(app.ID, app.MemberID, codes, reasons, "", report.FICOScore, 0), nil
	}

	// ── Stage 5: Model scoring ──────────────────────────────────────────────
	scoreResp, err := e.scoringClient.Score(ctx, e.buildScoringRequest(app, report, activeLCLoans))
	if err != nil {
		return nil, fmt.Errorf("scoring service unavailable: %w", err)
	}
	log.Info("model score received",
		zap.String("risk_band", scoreResp.RiskBand),
		zap.Float64("pd", scoreResp.PDEstimate))
	e.metrics.RecordHistogram("lee.score.pd_estimate", scoreResp.PDEstimate)

	// ── Stage 6: Post-score policy evaluation ───────────────────────────────
	if autoDenyBands[scoreResp.RiskBand] {
		e.metrics.Increment("lee.decline.auto_deny_band")
		return e.decline(app.ID, app.MemberID,
			[]DeclineCode{DeclineRiskBandAutoDeny},
			[]string{fmt.Sprintf("Risk band %s does not meet approval criteria", scoreResp.RiskBand)},
			scoreResp.RiskBand, report.FICOScore, scoreResp.PDEstimate), nil
	}

	// ── Stage 7: Manual review routing ─────────────────────────────────────
	if e.requiresManualReview(app, report, scoreResp) {
		e.metrics.Increment("lee.outcome.manual_review")
		d := &Decision{
			ApplicationID: app.ID,
			MemberID:      app.MemberID,
			Outcome:       OutcomeManualReview,
			RiskBand:      scoreResp.RiskBand,
			FICOScore:     report.FICOScore,
			PDEstimate:    scoreResp.PDEstimate,
			DecidedAt:     time.Now(),
			ModelVersion:  scoreResp.ModelVersion,
		}
		e.publishDecisionEvent(ctx, d)
		return d, nil
	}

	// ── Stage 8: Build loan offer ───────────────────────────────────────────
	offer, err := e.buildOffer(app.RequestedAmountCents, app.TermMonths,
		scoreResp.RiskBand, report.FICOScore)
	if err != nil {
		return nil, fmt.Errorf("offer construction failed: %w", err)
	}

	// ── Stage 9: Record and publish ────────────────────────────────────────
	e.metrics.Increment("lee.outcome.approved")
	d := &Decision{
		ApplicationID: app.ID,
		MemberID:      app.MemberID,
		Outcome:       OutcomeApproved,
		RiskBand:      scoreResp.RiskBand,
		FICOScore:     report.FICOScore,
		PDEstimate:    scoreResp.PDEstimate,
		Offer:         offer,
		DecidedAt:     time.Now(),
		ModelVersion:  scoreResp.ModelVersion,
	}
	if err := e.loanRepository.RecordDecision(ctx, d); err != nil {
		log.Error("failed to persist decision", zap.Error(err))
		// Non-fatal — decision event is still published
	}
	e.publishDecisionEvent(ctx, d)
	return d, nil
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

func (e *Engine) validateRequest(app *model.LoanApplication) ([]DeclineCode, []string) {
	var codes []DeclineCode
	var reasons []string

	add := func(c DeclineCode, r string) {
		codes = append(codes, c)
		reasons = append(reasons, r)
	}

	if app.RequestedAmountCents < minLoanAmountCents {
		add(DeclineAmountBelowMin, fmt.Sprintf("Requested amount $%.2f is below minimum $%.2f",
			float64(app.RequestedAmountCents)/100, float64(minLoanAmountCents)/100))
	}
	if app.RequestedAmountCents > maxLoanAmountCents {
		add(DeclineAmountAboveMax, fmt.Sprintf("Requested amount $%.2f exceeds maximum $%.2f",
			float64(app.RequestedAmountCents)/100, float64(maxLoanAmountCents)/100))
	}
	if !eligibleTerms[app.TermMonths] {
		add(DeclineInvalidTerm, fmt.Sprintf("Term %d months is not an eligible term", app.TermMonths))
	}
	if app.Applicant.AnnualIncomeCents <= 0 {
		add(DeclineInvalidIncome, "Annual income must be greater than zero")
	}
	if !isEligibleState(app.Applicant.State) {
		add(DeclineStateIneligible, fmt.Sprintf("LendingClub does not operate in %s", app.Applicant.State))
	}
	return codes, reasons
}

// ---------------------------------------------------------------------------
// Policy pre-screen
// ---------------------------------------------------------------------------

func (e *Engine) evaluatePolicyPreScreen(
	app *model.LoanApplication,
	report *bureau.Report,
	activeLCLoans int,
) ([]DeclineCode, []string) {
	var codes []DeclineCode
	var reasons []string

	add := func(c DeclineCode, r string) {
		codes = append(codes, c)
		reasons = append(reasons, r)
	}

	if report.FICOScore < minFICOScore {
		add(DeclineFICOBelowMin, fmt.Sprintf("FICO score %d is below minimum %d", report.FICOScore, minFICOScore))
	}
	if report.CreditHistoryMonths < minCreditHistoryMonths {
		add(DeclineInsufficientHistory, fmt.Sprintf("Credit history of %d months is below required %d months",
			report.CreditHistoryMonths, minCreditHistoryMonths))
	}
	if report.Inquiries6Months > maxInquiries6Months {
		add(DeclineExcessiveInquiries, fmt.Sprintf("%d inquiries in last 6 months exceeds limit of %d",
			report.Inquiries6Months, maxInquiries6Months))
	}
	if activeLCLoans >= maxConcurrentLCLoans {
		add(DeclineMaxLCLoansExceeded, fmt.Sprintf("%d active LC loans meets or exceeds limit of %d",
			activeLCLoans, maxConcurrentLCLoans))
	}

	// Active bankruptcy (within lookback)
	lookbackDate := time.Now().AddDate(-bankruptcyLookbackYrs, 0, 0)
	for _, b := range report.Bankruptcies {
		if b.FilingDate.After(lookbackDate) {
			add(DeclineActiveBankruptcy, fmt.Sprintf("Bankruptcy filed %s is within %d-year lookback",
				b.FilingDate.Format("2006-01-02"), bankruptcyLookbackYrs))
			break
		}
	}

	// Serious delinquency in lookback window
	seriousDelinqCutoff := time.Now().AddDate(0, 0, -seriousDelinqLookback)
	for _, t := range report.Tradelines {
		if t.WorstStatus12Months >= seriousDelinqDPD && t.LastStatusDate.After(seriousDelinqCutoff) {
			add(DeclineSeriousDelinquency, fmt.Sprintf("90+ DPD delinquency on account ending %s within 12 months",
				t.AccountNumberLast4))
			break
		}
	}

	// DTI hard cutoff
	dti, err := e.calculateDTI(app, report)
	if err != nil {
		e.logger.Warn("DTI calculation error; treating as cutoff exceeded", zap.Error(err))
		add(DeclineDTIExceedsHardCutoff, "DTI could not be calculated; treated as exceeding maximum")
	} else if dti > maxDTI {
		add(DeclineDTIExceedsHardCutoff, fmt.Sprintf("DTI of %.3f exceeds maximum %.2f", dti, maxDTI))
	}

	return codes, reasons
}

// ---------------------------------------------------------------------------
// DTI calculation
// ---------------------------------------------------------------------------

// calculateDTI computes debt-to-income ratio per CP-PL-2023-07 §4.1.
// Numerator: monthly bureau obligations + estimated new payment.
// Denominator: gross monthly income (verified or stated).
func (e *Engine) calculateDTI(app *model.LoanApplication, report *bureau.Report) (float64, error) {
	if app.Applicant.AnnualIncomeCents <= 0 {
		return 0, errors.New("annual income is zero or negative")
	}
	monthlyIncome := float64(app.Applicant.AnnualIncomeCents) / 12.0

	var monthlyObligations float64
	for _, t := range report.Tradelines {
		if t.Status != "OPEN" {
			continue
		}
		switch t.Type {
		case "REVOLVING":
			// 3% of revolving balance as monthly minimum (policy standard)
			monthlyObligations += float64(t.BalanceCents) * 0.03
		case "INSTALLMENT":
			monthlyObligations += float64(t.ScheduledMonthlyPaymentCents)
		case "MORTGAGE":
			monthlyObligations += float64(t.ScheduledMonthlyPaymentCents)
		}
	}

	// Estimate new loan monthly payment at stress rate (18% APR)
	newPayment := monthlyPayment(float64(app.RequestedAmountCents), app.TermMonths, 0.18)
	totalDebt := monthlyObligations + newPayment

	return totalDebt / monthlyIncome, nil
}

// ---------------------------------------------------------------------------
// Offer construction
// ---------------------------------------------------------------------------

func (e *Engine) buildOffer(amountCents int64, termMonths int, riskBand string, fico int) (*LoanOffer, error) {
	margin, ok := riskMarginBPS[riskBand]
	if !ok {
		return nil, fmt.Errorf("unknown risk band: %s", riskBand)
	}

	totalRateBPS := e.primeRateBPS + margin
	annualRate := float64(totalRateBPS) / 10000.0

	monthlyPmt := monthlyPayment(float64(amountCents), termMonths, annualRate)
	totalPaid := monthlyPmt * float64(termMonths)
	totalInterest := totalPaid - float64(amountCents)

	// Origination fee: tiered by risk band (100–600 bps of principal)
	origFeeBPS := originationFeeBPS(riskBand)
	origFeeCents := int64(float64(amountCents) * float64(origFeeBPS) / 10000.0)

	// Effective APR (includes origination fee amortized over term)
	effectiveAPRBPS := effectiveAPR(float64(amountCents), float64(origFeeCents),
		monthlyPmt, termMonths)

	return &LoanOffer{
		OfferedAmountCents:  amountCents,
		TermMonths:          termMonths,
		AnnualRateBPS:       totalRateBPS,
		MonthlyPaymentCents: int64(math.Round(monthlyPmt)),
		OriginationFeeBPS:   origFeeBPS,
		OriginationFeeCents: origFeeCents,
		TotalInterestCents:  int64(math.Round(totalInterest)),
		EffectiveAPRBPS:     effectiveAPRBPS,
		OfferExpiresAt:      time.Now().Add(e.offerTTL),
	}, nil
}

// monthlyPayment computes the standard amortized monthly payment.
func monthlyPayment(principalCents float64, termMonths int, annualRate float64) float64 {
	r := annualRate / 12.0
	if r == 0 {
		return principalCents / float64(termMonths)
	}
	return principalCents * r / (1 - math.Pow(1+r, -float64(termMonths)))
}

// effectiveAPR approximates the APR including the origination fee
// using Newton's method on the IRR of the cash flow stream.
func effectiveAPR(principal, feeCents, monthlyPmt float64, termMonths int) int {
	net := principal - feeCents
	// IRR solve: net = sum of pmt/(1+r)^t for t = 1..termMonths
	r := 0.01 // initial guess: 1% monthly
	for i := 0; i < 100; i++ {
		f := -net
		df := 0.0
		for t := 1; t <= termMonths; t++ {
			disc := math.Pow(1+r, -float64(t))
			f += monthlyPmt * disc
			df -= float64(t) * monthlyPmt * disc / (1 + r)
		}
		delta := f / df
		r -= delta
		if math.Abs(delta) < 1e-9 {
			break
		}
	}
	annualAPR := r * 12
	return int(math.Round(annualAPR * 10000)) // return as BPS
}

func originationFeeBPS(riskBand string) int {
	fees := map[string]int{
		"A1": 100, "A2": 150, "A3": 200,
		"B1": 250, "B2": 300, "B3": 350,
		"C1": 400, "C2": 450, "C3": 500,
		"D":  550, "E":  575, "F":  600,
		"G":  600,
	}
	if f, ok := fees[riskBand]; ok {
		return f
	}
	return 600
}

// ---------------------------------------------------------------------------
// Manual review routing
// ---------------------------------------------------------------------------

func (e *Engine) requiresManualReview(
	app *model.LoanApplication,
	report *bureau.Report,
	score *scoring.Response,
) bool {
	if manualReviewBands[score.RiskBand] {
		return true
	}
	// High income unverified above threshold ($150k)
	if app.Applicant.AnnualIncomeCents > 15_000_000 && !app.IncomeVerified {
		return true
	}
	// Soft fraud signal from scoring model
	if score.FraudScore > 0.65 {
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Scoring request builder
// ---------------------------------------------------------------------------

func (e *Engine) buildScoringRequest(
	app *model.LoanApplication,
	report *bureau.Report,
	activeLCLoans int,
) *scoring.Request {
	return &scoring.Request{
		ApplicationID:          app.ID,
		FICOScore:              report.FICOScore,
		VantageScore:           report.VantageScore,
		AnnualIncomeCents:      app.Applicant.AnnualIncomeCents,
		IncomeVerified:         app.IncomeVerified,
		RequestedAmountCents:   app.RequestedAmountCents,
		TermMonths:             app.TermMonths,
		LoanPurpose:            app.LoanPurpose,
		EmploymentStatus:       app.Applicant.EmploymentStatus,
		EmploymentMonths:       app.Applicant.EmploymentMonths,
		HomeOwnership:          app.Applicant.HomeOwnership,
		StateCode:              app.Applicant.State,
		RevolvingUtilization:   report.RevolvingUtilization,
		NumOpenAccounts:        report.NumOpenAccounts,
		Inquiries6Months:       report.Inquiries6Months,
		MthsSinceLastDelinq:    report.MonthsSinceLastDelinquency,
		MthsSinceLastDerog:     report.MonthsSinceLastDerogatory,
		NumDerogatoryMarks:     report.NumDerogatoryMarks,
		NumBankruptcies:        len(report.Bankruptcies),
		Collections12Mo:        report.Collections12MonthsExMedical,
		Chargeoffs24Mo:         report.Chargeoffs24Months,
		ActiveLCLoans:          activeLCLoans,
		EarliestCreditLineDate: report.EarliestCreditLineDate,
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func (e *Engine) decline(appID, memberID string, codes []DeclineCode, reasons []string,
	riskBand string, fico int, pd float64) *Decision {
	d := &Decision{
		ApplicationID:  appID,
		MemberID:       memberID,
		Outcome:        OutcomeDeclined,
		DeclineCodes:   codes,
		DeclineReasons: reasons,
		RiskBand:       riskBand,
		FICOScore:      fico,
		PDEstimate:     pd,
		DecidedAt:      time.Now(),
	}
	e.metrics.Increment("lee.outcome.declined")
	e.publishDecisionEvent(context.Background(), d)
	return d
}

func (e *Engine) publishDecisionEvent(ctx context.Context, d *Decision) {
	if err := e.eventPublisher.Publish(ctx, "loan.decisions", d); err != nil {
		e.logger.Error("failed to publish decision event",
			zap.String("application_id", d.ApplicationID),
			zap.Error(err))
		// Non-fatal; downstream consumers will reconcile via CDC
	}
}

// isEligibleState returns true if LendingClub currently accepts applications
// from the given two-letter state code.
// Source: Compliance Operations — updated monthly.
func isEligibleState(state string) bool {
	ineligible := map[string]bool{
		"IA": true, // Iowa — no active lending license
		"ID": true, // Idaho — pending license renewal
		"IN": true, // Indiana — rate cap incompatible with product
	}
	return !ineligible[state]
}

// SortDeclineCodes returns a deterministically ordered slice of decline codes,
// useful for consistent hashing and deduplication in downstream systems.
func SortDeclineCodes(codes []DeclineCode) []DeclineCode {
	strs := make([]string, len(codes))
	for i, c := range codes {
		strs[i] = string(c)
	}
	sort.Strings(strs)
	sorted := make([]DeclineCode, len(codes))
	for i, s := range strs {
		sorted[i] = DeclineCode(s)
	}
	return sorted
}
