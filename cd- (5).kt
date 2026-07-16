/*
 * ConsumerLoanRiskAssessment.kt
 */

package com.lendingclub.risk.consumer

import com.lendingclub.bureau.BureauClient
import com.lendingclub.bureau.model.BureauReport
import com.lendingclub.bureau.model.TradelineStatus
import com.lendingclub.bureau.model.TradelineType
import com.lendingclub.risk.model.*
import com.lendingclub.risk.scoring.ScoringClient
import com.lendingclub.risk.scoring.ScoringRequest
import com.lendingclub.risk.scoring.ScoringResponse
import io.micrometer.core.instrument.MeterRegistry
import kotlinx.coroutines.*
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.LocalDate
import java.time.Period
import java.time.temporal.ChronoUnit
import kotlin.math.ln
import kotlin.math.pow

// ─────────────────────────────────────────────────────────────────────────────
// Policy constants — Credit Policy CP-PL-2023-07
// ─────────────────────────────────────────────────────────────────────────────

private const val MIN_FICO                   = 580
private const val MAX_DTI                    = 0.65
private const val MIN_LOAN_AMOUNT_CENTS      = 100_000L    // $1,000
private const val MAX_LOAN_AMOUNT_CENTS      = 4_000_000L  // $40,000
private const val MIN_CREDIT_HISTORY_MONTHS  = 12
private const val MAX_CONCURRENT_LC_LOANS    = 2
private const val MAX_INQUIRIES_6MO          = 6
private const val BANKRUPTCY_LOOKBACK_YEARS  = 7L
private const val SERIOUS_DELINQ_DPD         = 90
private const val PRIME_RATE_BPS             = 850         // 8.50%

private val ELIGIBLE_TERMS = setOf(36, 60)

// Risk margin over prime rate (bps) per risk band — Pricing Policy PP-PL-2024-01
private val RISK_MARGIN_BPS = mapOf(
    "A1" to 120,  "A2" to 175,  "A3" to 230,
    "B1" to 295,  "B2" to 360,  "B3" to 430,
    "C1" to 510,  "C2" to 595,  "C3" to 685,
    "D"  to 790,  "E"  to 920,  "F"  to 1080,
    "G"  to 1290,
)

// Origination fee in bps of principal — tiered by risk band
private val ORIGINATION_FEE_BPS = mapOf(
    "A1" to 100, "A2" to 150, "A3" to 200,
    "B1" to 250, "B2" to 300, "B3" to 350,
    "C1" to 400, "C2" to 450, "C3" to 500,
    "D"  to 550, "E"  to 575, "F"  to 600, "G" to 600,
)

private val AUTO_DENY_BANDS    = setOf("G")
private val MANUAL_REVIEW_BANDS = setOf("E", "F")

// ─────────────────────────────────────────────────────────────────────────────
// Domain model
// ─────────────────────────────────────────────────────────────────────────────

enum class Outcome { APPROVED, DECLINED, MANUAL_REVIEW }

enum class DeclineCode(val description: String) {
    FICO_BELOW_MINIMUM            ("FICO score below minimum threshold"),
    DTI_EXCEEDS_HARD_CUTOFF       ("Debt-to-income ratio exceeds maximum"),
    ACTIVE_BANKRUPTCY             ("Active bankruptcy within lookback period"),
    RECENT_SERIOUS_DELINQUENCY    ("Serious delinquency within 12 months"),
    INSUFFICIENT_CREDIT_HISTORY   ("Credit history length insufficient"),
    EXCESSIVE_RECENT_INQUIRIES    ("Too many credit inquiries in last 6 months"),
    MAX_CONCURRENT_LC_LOANS       ("Maximum concurrent LendingClub loans reached"),
    AMOUNT_BELOW_MINIMUM          ("Requested amount below minimum"),
    AMOUNT_ABOVE_MAXIMUM          ("Requested amount exceeds maximum"),
    INVALID_LOAN_TERM             ("Loan term not offered"),
    IDENTITY_FAILED               ("Identity verification did not pass"),
    FRAUD_OFAC_FLAG               ("Pre-screen compliance check did not pass"),
    RISK_BAND_AUTO_DENY           ("Risk band does not meet approval criteria"),
    STATE_INELIGIBLE              ("LendingClub does not operate in applicant's state"),
    INVALID_INCOME                ("Reported income is invalid"),
}

data class LoanOffer(
    val offeredAmountCents:   Long,
    val termMonths:           Int,
    val annualRateBPS:        Int,
    val monthlyPaymentCents:  Long,
    val originationFeeBPS:    Int,
    val originationFeeCents:  Long,
    val totalInterestCents:   Long,
    val effectiveAPRBPS:      Int,
)

data class RiskAssessmentResult(
    val applicationId:  String,
    val memberId:       String,
    val outcome:        Outcome,
    val declineCodes:   List<DeclineCode> = emptyList(),
    val declineReasons: List<String>      = emptyList(),
    val riskBand:       String?           = null,
    val ficoScore:      Int?              = null,
    val pdEstimate:     Double?           = null,
    val offer:          LoanOffer?        = null,
    val modelVersion:   String?           = null,
    val assessedAt:     LocalDate         = LocalDate.now(),
)

// ─────────────────────────────────────────────────────────────────────────────
// Applicant risk profile — composite signal object used throughout assessment
// ─────────────────────────────────────────────────────────────────────────────

data class ApplicantRiskProfile(
    // Application signals
    val applicationId:        String,
    val memberId:             String,
    val requestedAmountCents: Long,
    val termMonths:           Int,
    val annualIncomeCents:    Long,
    val incomeVerified:       Boolean,
    val loanPurpose:          String,
    val employmentStatus:     String,
    val employmentMonths:     Int?,
    val homeOwnership:        String,
    val stateCode:            String,

    // Bureau-derived signals
    val ficoScore:                    Int,
    val vantageScore:                 Int?,
    val revolvingUtilization:         Double,
    val creditHistoryMonths:          Int,
    val numOpenAccounts:              Int,
    val inquiries6Months:             Int,
    val monthsSinceLastDelinq:        Int?,
    val monthsSinceLastDerog:         Int?,
    val numDerogatoryMarks:           Int,
    val numBankruptcies:              Int,
    val hasActiveBankruptcy:          Boolean,
    val hasSeriousDelinquency12Mo:    Boolean,
    val collections12MonthsExMedical: Int,
    val chargeoffs24Months:           Int,
    val monthlyBureauObligationsCents: Long,

    // Portfolio signals
    val activeLCLoans: Int,

    // Computed
    val dti:               Double,
    val creditAgeBucket:   String,   // e.g. "0-12", "13-24", "25-60", "60+"
) {
    /** Provides a loggable summary without PII. */
    fun toLogString() =
        "applicationId=$applicationId fico=$ficoScore riskBand=? " +
        "dti=${"%.3f".format(dti)} util=${"%.3f".format(revolvingUtilization)} " +
        "history=${creditHistoryMonths}mo activeLCLoans=$activeLCLoans"
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

@Service
class ConsumerLoanRiskAssessmentService(
    private val bureauClient:      BureauClient,
    private val scoringClient:     ScoringClient,
    private val applicationRepo:   ApplicationRepository,
    private val meterRegistry:     MeterRegistry,
    @Value("\${clra.income.verification.threshold.cents:7500000}")
    private val incomeVerificationThresholdCents: Long,
    @Value("\${clra.fraud.score.manual.review.threshold:0.65}")
    private val fraudScoreManualReviewThreshold: Double,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // ── Primary assessment entry point ──────────────────────────────────────

    /**
     * Assesses a consumer loan application end-to-end.
     *
     * Runs validation, bureau pull, profile construction, policy pre-screen,
     * model scoring, and offer construction concurrently where possible.
     *
     * @param request Validated loan application request.
     * @return [RiskAssessmentResult] with outcome, offer terms if approved,
     *         and structured decline codes if declined.
     */
    suspend fun assess(request: LoanApplicationRequest): RiskAssessmentResult =
        withContext(Dispatchers.IO) {
            val appId = request.applicationId
            val timer = meterRegistry.timer("clra.assess.duration").start()
            log.info("Starting risk assessment: applicationId={}", appId)

            try {
                // Step 1: Request validation
                val validationDeclines = validateRequest(request)
                if (validationDeclines.isNotEmpty()) {
                    return@withContext decline(appId, request.memberId, validationDeclines)
                }

                // Step 2: Parallel — bureau pull + active loan count
                val bureauDeferred  = async { bureauClient.pull(request.applicant) }
                val loanCountDeferred = async {
                    applicationRepo.countActiveLoans(request.applicant.ssn)
                }

                val bureauReport  = bureauDeferred.await()
                val activeLCLoans = loanCountDeferred.await()

                // Step 3: Build composite risk profile
                val profile = buildRiskProfile(request, bureauReport, activeLCLoans)
                log.info("Risk profile built: {}", profile.toLogString())

                // Step 4: Policy pre-screen
                val policyDeclines = evaluatePolicyPreScreen(profile)
                if (policyDeclines.isNotEmpty()) {
                    policyDeclines.forEach {
                        meterRegistry.counter("clra.decline", "reason", it.name).increment()
                    }
                    return@withContext decline(appId, request.memberId, policyDeclines, profile.ficoScore)
                }

                // Step 5: Model scoring
                val scoreResponse = scoringClient.score(buildScoringRequest(profile))
                log.info("Scoring complete: applicationId={} riskBand={} pd={}",
                    appId, scoreResponse.riskBand, scoreResponse.pdEstimate)
                meterRegistry.summary("clra.score.pd").record(scoreResponse.pdEstimate)

                // Step 6: Post-score policy overlay
                if (scoreResponse.riskBand in AUTO_DENY_BANDS) {
                    meterRegistry.counter("clra.decline", "reason", "AUTO_DENY_BAND").increment()
                    return@withContext decline(
                        appId, request.memberId,
                        listOf(DeclineCode.RISK_BAND_AUTO_DENY),
                        profile.ficoScore,
                        scoreResponse.riskBand,
                        scoreResponse.pdEstimate,
                    )
                }

                // Step 7: Manual review routing
                if (requiresManualReview(profile, scoreResponse)) {
                    meterRegistry.counter("clra.outcome", "outcome", "manual_review").increment()
                    return@withContext RiskAssessmentResult(
                        applicationId = appId,
                        memberId      = request.memberId,
                        outcome       = Outcome.MANUAL_REVIEW,
                        riskBand      = scoreResponse.riskBand,
                        ficoScore     = profile.ficoScore,
                        pdEstimate    = scoreResponse.pdEstimate,
                        modelVersion  = scoreResponse.modelVersion,
                    )
                }

                // Step 8: Build offer
                val offer = buildOffer(profile.requestedAmountCents, profile.termMonths,
                    scoreResponse.riskBand)
                meterRegistry.counter("clra.outcome", "outcome", "approved").increment()

                RiskAssessmentResult(
                    applicationId = appId,
                    memberId      = request.memberId,
                    outcome       = Outcome.APPROVED,
                    riskBand      = scoreResponse.riskBand,
                    ficoScore     = profile.ficoScore,
                    pdEstimate    = scoreResponse.pdEstimate,
                    offer         = offer,
                    modelVersion  = scoreResponse.modelVersion,
                )
            } finally {
                timer.stop()
            }
        }

    // ── Step 1: Validation ──────────────────────────────────────────────────

    private fun validateRequest(req: LoanApplicationRequest): List<DeclineCode> {
        val codes = mutableListOf<DeclineCode>()
        if (req.requestedAmountCents < MIN_LOAN_AMOUNT_CENTS)
            codes += DeclineCode.AMOUNT_BELOW_MINIMUM
        if (req.requestedAmountCents > MAX_LOAN_AMOUNT_CENTS)
            codes += DeclineCode.AMOUNT_ABOVE_MAXIMUM
        if (req.termMonths !in ELIGIBLE_TERMS)
            codes += DeclineCode.INVALID_LOAN_TERM
        if (req.applicant.annualIncomeCents <= 0)
            codes += DeclineCode.INVALID_INCOME
        if (!isEligibleState(req.applicant.state))
            codes += DeclineCode.STATE_INELIGIBLE
        return codes
    }

    // ── Step 3: Risk profile construction ──────────────────────────────────

    private fun buildRiskProfile(
        req: LoanApplicationRequest,
        report: BureauReport,
        activeLCLoans: Int,
    ): ApplicantRiskProfile {
        val monthlyIncome = req.applicant.annualIncomeCents / 12.0

        // Monthly obligations from bureau tradelines
        val bureauObligations = report.tradelines
            .filter { it.status == TradelineStatus.OPEN }
            .sumOf { tradeline ->
                when (tradeline.type) {
                    TradelineType.REVOLVING    -> tradeline.balanceCents * 0.03
                    TradelineType.INSTALLMENT,
                    TradelineType.MORTGAGE     -> tradeline.scheduledMonthlyPaymentCents.toDouble()
                    else                       -> 0.0
                }
            }

        // New loan estimated payment (stress rate 18% APR)
        val newPayment = monthlyPayment(
            req.requestedAmountCents.toDouble(), req.termMonths, 0.18
        )
        val dti = (bureauObligations + newPayment) / monthlyIncome.coerceAtLeast(1.0)

        // Active bankruptcy check
        val bankruptcyLookbackDate = LocalDate.now().minusYears(BANKRUPTCY_LOOKBACK_YEARS)
        val hasActiveBankruptcy = report.bankruptcies.any {
            it.filingDate.isAfter(bankruptcyLookbackDate)
        }

        // Serious delinquency (90+ DPD) in last 12 months
        val hasSeriousDelinq12Mo = report.tradelines.any {
            it.worstStatus12Months >= SERIOUS_DELINQ_DPD
        }

        val creditHistoryMonths = ChronoUnit.MONTHS.between(
            report.earliestCreditLineDate, LocalDate.now()
        ).toInt().coerceAtLeast(0)

        val creditAgeBucket = when {
            creditHistoryMonths in 0..12   -> "0-12"
            creditHistoryMonths in 13..24  -> "13-24"
            creditHistoryMonths in 25..60  -> "25-60"
            else                           -> "60+"
        }

        return ApplicantRiskProfile(
            applicationId            = req.applicationId,
            memberId                 = req.memberId,
            requestedAmountCents     = req.requestedAmountCents,
            termMonths               = req.termMonths,
            annualIncomeCents        = req.applicant.annualIncomeCents,
            incomeVerified           = req.incomeVerified,
            loanPurpose              = req.loanPurpose,
            employmentStatus         = req.applicant.employmentStatus,
            employmentMonths         = req.applicant.employmentMonths,
            homeOwnership            = req.applicant.homeOwnership,
            stateCode                = req.applicant.state,
            ficoScore                = report.ficoScore,
            vantageScore             = report.vantageScore,
            revolvingUtilization     = report.revolvingUtilization,
            creditHistoryMonths      = creditHistoryMonths,
            numOpenAccounts          = report.numOpenAccounts,
            inquiries6Months         = report.inquiries6Months,
            monthsSinceLastDelinq    = report.monthsSinceLastDelinquency,
            monthsSinceLastDerog     = report.monthsSinceLastDerogatory,
            numDerogatoryMarks       = report.numDerogatoryMarks,
            numBankruptcies          = report.bankruptcies.size,
            hasActiveBankruptcy      = hasActiveBankruptcy,
            hasSeriousDelinquency12Mo = hasSeriousDelinq12Mo,
            collections12MonthsExMedical = report.collections12MonthsExMedical,
            chargeoffs24Months       = report.chargeoffs24Months,
            monthlyBureauObligationsCents = bureauObligations.toLong(),
            activeLCLoans            = activeLCLoans,
            dti                      = dti,
            creditAgeBucket          = creditAgeBucket,
        )
    }

    // ── Step 4: Policy pre-screen ───────────────────────────────────────────

    private fun evaluatePolicyPreScreen(profile: ApplicantRiskProfile): List<DeclineCode> {
        val codes = mutableListOf<DeclineCode>()
        if (profile.ficoScore < MIN_FICO)
            codes += DeclineCode.FICO_BELOW_MINIMUM
        if (profile.creditHistoryMonths < MIN_CREDIT_HISTORY_MONTHS)
            codes += DeclineCode.INSUFFICIENT_CREDIT_HISTORY
        if (profile.inquiries6Months > MAX_INQUIRIES_6MO)
            codes += DeclineCode.EXCESSIVE_RECENT_INQUIRIES
        if (profile.activeLCLoans >= MAX_CONCURRENT_LC_LOANS)
            codes += DeclineCode.MAX_CONCURRENT_LC_LOANS
        if (profile.hasActiveBankruptcy)
            codes += DeclineCode.ACTIVE_BANKRUPTCY
        if (profile.hasSeriousDelinquency12Mo)
            codes += DeclineCode.RECENT_SERIOUS_DELINQUENCY
        if (profile.dti > MAX_DTI)
            codes += DeclineCode.DTI_EXCEEDS_HARD_CUTOFF
        return codes
    }

    // ── Step 7: Manual review ───────────────────────────────────────────────

    private fun requiresManualReview(
        profile: ApplicantRiskProfile,
        score: ScoringResponse,
    ): Boolean {
        if (score.riskBand in MANUAL_REVIEW_BANDS) return true
        if (profile.annualIncomeCents > incomeVerificationThresholdCents && !profile.incomeVerified) return true
        if ((score.fraudScore ?: 0.0) > fraudScoreManualReviewThreshold) return true
        return false
    }

    // ── Step 8: Offer construction ──────────────────────────────────────────

    private fun buildOffer(amountCents: Long, termMonths: Int, riskBand: String): LoanOffer {
        val marginBPS = RISK_MARGIN_BPS[riskBand]
            ?: error("Unknown risk band: $riskBand")
        val totalRateBPS = PRIME_RATE_BPS + marginBPS
        val annualRate   = totalRateBPS / 10_000.0

        val monthlyPmt   = monthlyPayment(amountCents.toDouble(), termMonths, annualRate)
        val totalPaid    = monthlyPmt * termMonths
        val totalInterest = totalPaid - amountCents

        val origFeeBPS   = ORIGINATION_FEE_BPS[riskBand] ?: 600
        val origFeeCents = (amountCents * origFeeBPS / 10_000.0).toLong()

        val effectiveAPR = effectiveAPRBPS(amountCents.toDouble(), origFeeCents.toDouble(),
            monthlyPmt, termMonths)

        return LoanOffer(
            offeredAmountCents  = amountCents,
            termMonths          = termMonths,
            annualRateBPS       = totalRateBPS,
            monthlyPaymentCents = monthlyPmt.toLong(),
            originationFeeBPS   = origFeeBPS,
            originationFeeCents = origFeeCents,
            totalInterestCents  = totalInterest.toLong(),
            effectiveAPRBPS     = effectiveAPR,
        )
    }

    // ── Scoring request builder ─────────────────────────────────────────────

    private fun buildScoringRequest(p: ApplicantRiskProfile) = ScoringRequest(
        applicationId            = p.applicationId,
        ficoScore                = p.ficoScore,
        vantageScore             = p.vantageScore,
        annualIncomeCents        = p.annualIncomeCents,
        incomeVerified           = p.incomeVerified,
        requestedAmountCents     = p.requestedAmountCents,
        termMonths               = p.termMonths,
        loanPurpose              = p.loanPurpose,
        employmentStatus         = p.employmentStatus,
        employmentMonths         = p.employmentMonths,
        homeOwnership            = p.homeOwnership,
        stateCode                = p.stateCode,
        revolvingUtilization     = p.revolvingUtilization,
        numOpenAccounts          = p.numOpenAccounts,
        inquiries6Months         = p.inquiries6Months,
        monthsSinceLastDelinq    = p.monthsSinceLastDelinq,
        monthsSinceLastDerog     = p.monthsSinceLastDerog,
        numDerogatoryMarks       = p.numDerogatoryMarks,
        numBankruptcies          = p.numBankruptcies,
        collections12Mo          = p.collections12MonthsExMedical,
        chargeoffs24Mo           = p.chargeoffs24Months,
        activeLCLoans            = p.activeLCLoans,
    )

    // ── Decline helper ──────────────────────────────────────────────────────

    private fun decline(
        appId: String, memberId: String,
        codes: List<DeclineCode>,
        ficoScore: Int? = null,
        riskBand: String? = null,
        pd: Double? = null,
    ) = RiskAssessmentResult(
        applicationId  = appId,
        memberId       = memberId,
        outcome        = Outcome.DECLINED,
        declineCodes   = codes,
        declineReasons = codes.map { it.description },
        ficoScore      = ficoScore,
        riskBand       = riskBand,
        pdEstimate     = pd,
    )

    // ── Finance math helpers ────────────────────────────────────────────────

    private fun monthlyPayment(principalCents: Double, termMonths: Int, annualRate: Double): Double {
        val r = annualRate / 12.0
        return if (r == 0.0) principalCents / termMonths
        else principalCents * r / (1 - (1 + r).pow(-termMonths.toDouble()))
    }

    /**
     * Approximates effective APR in BPS including the origination fee,
     * using 50 iterations of Newton's method on the loan cash flow IRR.
     */
    private fun effectiveAPRBPS(
        principal: Double, feeCents: Double, monthlyPmt: Double, termMonths: Int,
    ): Int {
        val net = principal - feeCents
        var r   = 0.01 // 1% monthly starting guess
        repeat(50) {
            var f  = -net
            var df = 0.0
            for (t in 1..termMonths) {
                val disc = (1 + r).pow(-t.toDouble())
                f  += monthlyPmt * disc
                df -= t * monthlyPmt * disc / (1 + r)
            }
            r -= f / df
        }
        return (r * 12 * 10_000).toInt() // monthly IRR → annual APR → BPS
    }

    private fun isEligibleState(state: String): Boolean {
        val ineligible = setOf("IA", "ID", "IN")
        return state !in ineligible
    }
}
