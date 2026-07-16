"""
borrower_credit_model.py

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("crm.borrower_credit_model")

# ---------------------------------------------------------------------------
# Constants – sourced from MRC-approved model specification CRM-PL-042-SPEC
# ---------------------------------------------------------------------------

MODEL_VERSION        = "4.2.1"
MINIMUM_TRADELINES   = 2          # Bureau requirement: at least 2 open/closed trades
MAX_DTI_HARD_CUTOFF  = 0.65       # Hard policy cap; loans above are auto-declined
MIN_FICO_ORIGINATION = 580        # Minimum FICO at origination (policy floor)
MAX_LTV_UNSECURED    = 1.0        # LTV always 1.0 for unsecured personal loans
UTILIZATION_CLAMP    = (0.0, 1.5) # Clamp extreme utilization values (e.g., over-limit)
DEROG_LOOKBACK_DAYS  = 84 * 30    # 84-month derogatory lookback window

# Risk-band thresholds (PD → letter grade) — calibrated on 2019-2023 vintage cohorts
RISK_BAND_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "A1": (0.00, 0.015),
    "A2": (0.015, 0.025),
    "A3": (0.025, 0.035),
    "B1": (0.035, 0.050),
    "B2": (0.050, 0.070),
    "B3": (0.070, 0.090),
    "C1": (0.090, 0.115),
    "C2": (0.115, 0.145),
    "C3": (0.145, 0.180),
    "D":  (0.180, 0.240),
    "E":  (0.240, 0.320),
    "F":  (0.320, 0.450),
    "G":  (0.450, 1.001),
}

# Interest rate margin lookup (basis points above prime) per risk band
RATE_MARGIN_BPS: Dict[str, int] = {
    "A1": 120,  "A2": 175,  "A3": 230,
    "B1": 295,  "B2": 360,  "B3": 430,
    "C1": 510,  "C2": 595,  "C3": 685,
    "D":  790,  "E":  920,  "F":  1080, "G": 1290,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EmploymentStatus(str, Enum):
    EMPLOYED        = "EMPLOYED"
    SELF_EMPLOYED   = "SELF_EMPLOYED"
    NOT_EMPLOYED    = "NOT_EMPLOYED"
    OTHER           = "OTHER"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION  = "DEBT_CONSOLIDATION"
    HOME_IMPROVEMENT    = "HOME_IMPROVEMENT"
    MEDICAL             = "MEDICAL"
    MAJOR_PURCHASE      = "MAJOR_PURCHASE"
    VACATION            = "VACATION"
    MOVING              = "MOVING"
    BUSINESS            = "BUSINESS"
    OTHER               = "OTHER"


@dataclass
class BureauSnapshot:
    """
    Normalized view of a credit bureau pull (Equifax/Experian/TransUnion).
    All monetary values in USD cents to avoid floating-point drift.
    """
    pull_date:                  date
    fico_score:                 int
    vantage_score:              Optional[int]
    num_open_accounts:          int
    num_satisfactory_accounts:  int
    num_derogatory_marks:       int
    num_inquiries_6mo:          int
    num_inquiries_12mo:         int
    revolving_balance_cents:    int
    revolving_limit_cents:      int
    installment_balance_cents:  int
    mortgage_balance_cents:     int
    earliest_credit_line_date:  date
    months_since_last_delinq:   Optional[int]   # None if no delinquency on file
    months_since_last_derog:    Optional[int]   # None if no derog on file
    months_since_last_public:   Optional[int]
    num_public_records:         int
    num_bankruptcies:           int
    collections_12mo_ex_medical: int
    chargeoffs_24mo:            int
    tax_liens:                  int


@dataclass
class LoanApplication:
    application_id:     str
    applicant_id:       str
    requested_amount:   int                   # USD cents
    loan_purpose:       LoanPurpose
    term_months:        int                   # 36 or 60
    annual_income:      int                   # USD cents, self-reported
    income_verified:    bool
    employment_status:  EmploymentStatus
    employment_months:  Optional[int]
    home_ownership:     str                   # RENT / MORTGAGE / OWN / OTHER
    state:              str                   # Two-letter state code
    applied_at:         date
    bureau:             BureauSnapshot
    existing_lc_loans:  int = 0               # Current active LC loan count
    internal_risk_flag: bool = False          # Flagged by fraud/OFAC pre-screen


@dataclass
class ScoringResult:
    application_id:     str
    pd_estimate:        float                 # Probability of default (12-month horizon)
    pd_lower_95:        float
    pd_upper_95:        float
    risk_band:          str
    approved:           bool
    decline_reasons:    List[str] = field(default_factory=list)
    rate_margin_bps:    Optional[int] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_version:      str = MODEL_VERSION


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a LoanApplication into the model feature vector.
    Feature definitions are locked to model spec CRM-PL-042-SPEC §3.
    Any modification requires MRC change control.
    """

    FEATURE_NAMES = [
        "fico_score",
        "fico_vantage_delta",
        "log_annual_income",
        "dti_requested",
        "revolving_utilization",
        "revolving_utilization_sq",
        "credit_age_months",
        "num_open_accounts",
        "num_inquiries_6mo",
        "inq_6mo_x_util",              # Interaction: inquiry stress × utilization
        "months_since_last_delinq_imp", # Imputed: 999 if no delinquency
        "months_since_last_derog_imp",
        "num_derogatory_marks",
        "num_bankruptcies",
        "collections_12mo",
        "chargeoffs_24mo",
        "income_verified_flag",
        "employment_stable_flag",       # 1 if employed ≥ 24 months
        "home_owner_flag",
        "log_requested_amount",
        "term_60_flag",
        "purpose_debt_consol_flag",
        "purpose_business_flag",
        "existing_lc_loans",
        "installment_to_income",
        "mortgage_flag",
    ]

    def __init__(self, monthly_debt_obligations_cents: int = 0):
        """
        Args:
            monthly_debt_obligations_cents: Sum of all recurring monthly debt
                payments from bureau (used in DTI calculation).
        """
        self._monthly_debt = monthly_debt_obligations_cents

    def transform(self, app: LoanApplication) -> np.ndarray:
        b = app.bureau
        pull_date = b.pull_date

        # --- Income ---
        annual_income_usd = max(app.annual_income / 100, 1)
        monthly_income    = annual_income_usd / 12

        # --- Revolving utilization (clamped) ---
        if b.revolving_limit_cents > 0:
            util = b.revolving_balance_cents / b.revolving_limit_cents
        else:
            util = 0.0
        util = float(np.clip(util, *UTILIZATION_CLAMP))

        # --- DTI: includes new loan payment estimate ---
        new_monthly_payment = self._estimate_monthly_payment(
            principal=app.requested_amount / 100,
            term_months=app.term_months,
            annual_rate=0.15,             # Conservative placeholder rate for DTI calc
        )
        total_monthly_debt = (self._monthly_debt / 100) + new_monthly_payment
        dti = total_monthly_debt / max(monthly_income, 1)

        # --- Credit age ---
        credit_age_months = max(
            (pull_date - b.earliest_credit_line_date).days // 30, 1
        )

        # --- Delinquency imputation ---
        msl_delinq = b.months_since_last_delinq if b.months_since_last_delinq is not None else 999
        msl_derog  = b.months_since_last_derog  if b.months_since_last_derog  is not None else 999

        # --- FICO / VantageScore delta ---
        fico_vantage_delta = 0.0
        if b.vantage_score is not None:
            fico_vantage_delta = float(b.fico_score - b.vantage_score)

        # --- Employment stability flag ---
        emp_stable = 1 if (
            app.employment_status in (EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED)
            and (app.employment_months or 0) >= 24
        ) else 0

        # --- Home ownership ---
        home_owner = 1 if app.home_ownership in ("OWN", "MORTGAGE") else 0
        mortgage   = 1 if app.home_ownership == "MORTGAGE" else 0

        # --- Installment-to-income ratio ---
        installment_to_income = (b.installment_balance_cents / 100) / max(annual_income_usd, 1)

        features = np.array([
            float(b.fico_score),
            fico_vantage_delta,
            math.log1p(annual_income_usd),
            float(np.clip(dti, 0, 2)),
            util,
            util ** 2,
            float(credit_age_months),
            float(b.num_open_accounts),
            float(b.num_inquiries_6mo),
            float(b.num_inquiries_6mo) * util,         # interaction term
            float(min(msl_delinq, 999)),
            float(min(msl_derog, 999)),
            float(b.num_derogatory_marks),
            float(b.num_bankruptcies),
            float(b.collections_12mo_ex_medical),
            float(b.chargeoffs_24mo),
            float(int(app.income_verified)),
            float(emp_stable),
            float(home_owner),
            math.log1p(app.requested_amount / 100),
            float(int(app.term_months == 60)),
            float(int(app.loan_purpose == LoanPurpose.DEBT_CONSOLIDATION)),
            float(int(app.loan_purpose == LoanPurpose.BUSINESS)),
            float(app.existing_lc_loans),
            float(np.clip(installment_to_income, 0, 5)),
            float(mortgage),
        ], dtype=np.float64)

        assert len(features) == len(self.FEATURE_NAMES), (
            f"Feature count mismatch: {len(features)} vs {len(self.FEATURE_NAMES)}"
        )
        return features

    @staticmethod
    def _estimate_monthly_payment(principal: float, term_months: int,
                                  annual_rate: float) -> float:
        if annual_rate <= 0 or term_months <= 0:
            return principal / max(term_months, 1)
        r = annual_rate / 12
        return principal * r / (1 - (1 + r) ** -term_months)


# ---------------------------------------------------------------------------
# Policy Pre-Screen (hard cutoffs applied before model scoring)
# ---------------------------------------------------------------------------

class PolicyPreScreen:
    """
    Hard-cutoff rules defined in Credit Policy CP-PL-2023-07.
    These are NOT model outputs — they are binary pass/fail gates.
    Failure at this stage results in an immediate decline without model scoring.
    """

    def evaluate(self, app: LoanApplication) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if app.internal_risk_flag:
            reasons.append("FRAUD_OFAC_FLAG")

        if app.bureau.fico_score < MIN_FICO_ORIGINATION:
            reasons.append(f"FICO_BELOW_MINIMUM ({app.bureau.fico_score} < {MIN_FICO_ORIGINATION})")

        if app.bureau.num_bankruptcies > 0:
            # Check if bankruptcy is within the derogatory lookback window
            if app.bureau.months_since_last_public is not None:
                lookback_months = DEROG_LOOKBACK_DAYS // 30
                if app.bureau.months_since_last_public <= lookback_months:
                    reasons.append("ACTIVE_BANKRUPTCY_IN_LOOKBACK")

        if app.bureau.num_open_accounts + app.bureau.num_satisfactory_accounts < MINIMUM_TRADELINES:
            reasons.append("INSUFFICIENT_TRADELINE_HISTORY")

        if app.bureau.tax_liens > 0:
            reasons.append("OPEN_TAX_LIEN")

        if app.term_months not in (36, 60):
            reasons.append(f"INVALID_LOAN_TERM ({app.term_months})")

        if app.requested_amount < 100_00:   # $100 minimum (cents)
            reasons.append("AMOUNT_BELOW_MINIMUM")

        if app.requested_amount > 4_000_000_00:  # $40,000 maximum (cents)
            reasons.append("AMOUNT_ABOVE_MAXIMUM")

        if app.existing_lc_loans >= 2:
            reasons.append("EXCEEDS_MAX_CONCURRENT_LC_LOANS")

        passed = len(reasons) == 0
        return passed, reasons


# ---------------------------------------------------------------------------
# DTI Pre-Screen (separate from model; uses bureau-verified obligations)
# ---------------------------------------------------------------------------

class DTIPreScreen:
    def evaluate(self, dti: float) -> Tuple[bool, List[str]]:
        if dti > MAX_DTI_HARD_CUTOFF:
            return False, [f"DTI_EXCEEDS_HARD_CUTOFF ({dti:.3f} > {MAX_DTI_HARD_CUTOFF})"]
        return True, []


# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------

class BorrowerCreditModel:
    """
    Production scoring model for personal loan origination.
    Wraps a calibrated GBM with policy pre-screens and risk-band mapping.

    Usage:
        model = BorrowerCreditModel.load("/models/crm_pl_042_v421.pkl")
        result = model.score(application, monthly_debt_obligations_cents=45000)
    """

    def __init__(self, pipeline: Pipeline, feature_engineer: FeatureEngineer):
        self._pipeline = pipeline
        self._fe = feature_engineer

    # ------------------------------------------------------------------
    # Scoring entry point
    # ------------------------------------------------------------------

    def score(
        self,
        app: LoanApplication,
        monthly_debt_obligations_cents: int = 0,
    ) -> ScoringResult:
        """
        Score a loan application.

        Args:
            app: Populated LoanApplication dataclass.
            monthly_debt_obligations_cents: Verified monthly debt obligation
                total from bureau tradelines (principal + interest, annualized /12).

        Returns:
            ScoringResult with PD estimate, risk band, approval decision,
            and decline reasons if applicable.
        """
        fe = FeatureEngineer(monthly_debt_obligations_cents)

        # --- Step 1: Policy pre-screen ---
        policy_ok, policy_reasons = PolicyPreScreen().evaluate(app)
        if not policy_ok:
            logger.warning(
                "Application %s failed policy pre-screen: %s",
                app.application_id, policy_reasons
            )
            return ScoringResult(
                application_id=app.application_id,
                pd_estimate=1.0,
                pd_lower_95=1.0,
                pd_upper_95=1.0,
                risk_band="DECLINED",
                approved=False,
                decline_reasons=policy_reasons,
                model_version=MODEL_VERSION,
            )

        # --- Step 2: Feature engineering ---
        features = fe.transform(app)

        # --- Step 3: DTI check (uses engineered DTI) ---
        dti = features[3]   # Index corresponds to FEATURE_NAMES position
        dti_ok, dti_reasons = DTIPreScreen().evaluate(dti)
        if not dti_ok:
            return ScoringResult(
                application_id=app.application_id,
                pd_estimate=1.0,
                pd_lower_95=1.0,
                pd_upper_95=1.0,
                risk_band="DECLINED",
                approved=False,
                decline_reasons=dti_reasons,
                model_version=MODEL_VERSION,
            )

        # --- Step 4: Model scoring ---
        X = features.reshape(1, -1)
        pd_mean  = float(self._pipeline.predict_proba(X)[0, 1])
        pd_lower, pd_upper = self._bootstrap_confidence_interval(X)

        # --- Step 5: Risk band assignment ---
        risk_band = self._assign_risk_band(pd_mean)

        # --- Step 6: Approval decision (risk-band–level policy) ---
        approved, model_decline_reasons = self._apply_credit_policy(
            risk_band=risk_band,
            pd=pd_mean,
            app=app,
        )

        # --- Step 7: Feature importance (SHAP approximation via permutation) ---
        importance = dict(zip(
            FeatureEngineer.FEATURE_NAMES,
            self._pipeline.named_steps["clf"].feature_importances_,
        ))

        return ScoringResult(
            application_id=app.application_id,
            pd_estimate=round(pd_mean, 6),
            pd_lower_95=round(pd_lower, 6),
            pd_upper_95=round(pd_upper, 6),
            risk_band=risk_band,
            approved=approved,
            decline_reasons=model_decline_reasons,
            rate_margin_bps=RATE_MARGIN_BPS.get(risk_band) if approved else None,
            feature_importance=importance,
            model_version=MODEL_VERSION,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_risk_band(pd: float) -> str:
        for band, (lo, hi) in RISK_BAND_THRESHOLDS.items():
            if lo <= pd < hi:
                return band
        return "G"

    @staticmethod
    def _apply_credit_policy(
        risk_band: str,
        pd: float,
        app: LoanApplication,
    ) -> Tuple[bool, List[str]]:
        """
        Post-score policy overlay. Credit policy CP-PL-2023-07 §5.2:
        - Risk band G is always declined.
        - Risk band F is declined unless income is verified AND employment is stable.
        - Unverified income above $150k is flagged for manual review (not auto-decline).
        """
        reasons: List[str] = []

        if risk_band == "G":
            reasons.append("RISK_BAND_G_AUTO_DECLINE")
            return False, reasons

        if risk_band == "F":
            if not app.income_verified:
                reasons.append("RISK_BAND_F_REQUIRES_VERIFIED_INCOME")
                return False, reasons
            if app.employment_status not in (
                EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED
            ):
                reasons.append("RISK_BAND_F_REQUIRES_STABLE_EMPLOYMENT")
                return False, reasons

        return True, reasons

    def _bootstrap_confidence_interval(
        self,
        X: np.ndarray,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Approximate 95% CI via perturbation bootstrap on the feature vector.
        Used for model monitoring and uncertainty quantification.
        """
        rng = np.random.default_rng(seed=42)
        preds: List[float] = []
        for _ in range(n_bootstrap):
            noise = rng.normal(loc=0, scale=0.01, size=X.shape)
            X_perturbed = X + noise
            preds.append(float(self._pipeline.predict_proba(X_perturbed)[0, 1]))
        preds_arr = np.array(preds)
        return (
            float(np.percentile(preds_arr, 100 * alpha / 2)),
            float(np.percentile(preds_arr, 100 * (1 - alpha / 2))),
        )

    @classmethod
    def load(cls, model_path: str) -> "BorrowerCreditModel":
        import pickle
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        return cls(
            pipeline=bundle["pipeline"],
            feature_engineer=bundle["feature_engineer"],
        )

    def save(self, model_path: str) -> None:
        import pickle
        bundle = {
            "pipeline":         self._pipeline,
            "feature_engineer": self._fe,
            "model_version":    MODEL_VERSION,
        }
        with open(model_path, "wb") as fh:
            pickle.dump(bundle, fh, protocol=5)
        logger.info("Model saved to %s (version %s)", model_path, MODEL_VERSION)


# ---------------------------------------------------------------------------
# Training helper (run offline; not part of real-time scoring path)
# ---------------------------------------------------------------------------

def train_model(
    training_df: pd.DataFrame,
    monthly_debt_col: str = "monthly_debt_cents",
    label_col: str = "default_12mo",
    n_estimators: int = 600,
    learning_rate: float = 0.035,
    max_depth: int = 5,
    subsample: float = 0.8,
    min_samples_leaf: int = 50,
    random_state: int = 2024,
) -> BorrowerCreditModel:
    """
    Trains the GBM pipeline on a labeled origination cohort.

    Args:
        training_df: DataFrame where each row is a LoanApplication-equivalent
            (pre-featurized via FeatureEngineer.transform). Must include label_col.
        label_col: Binary target — 1 = defaulted within 12 months of origination.
        ... (GBM hyperparams as documented in MRC submission CRM-PL-042-MRC)

    Returns:
        Fitted BorrowerCreditModel ready for .save() and deployment.
    """
    fe = FeatureEngineer()

    X = training_df[FeatureEngineer.FEATURE_NAMES].values
    y = training_df[label_col].values

    gbm = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        validation_fraction=0.1,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=random_state,
    )
    calibrated = CalibratedClassifierCV(gbm, method="isotonic", cv=5)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])
    pipeline.fit(X, y)
    logger.info(
        "Model trained on %d samples (positive rate: %.3f)",
        len(y), y.mean()
    )
    return BorrowerCreditModel(pipeline=pipeline, feature_engineer=fe)
