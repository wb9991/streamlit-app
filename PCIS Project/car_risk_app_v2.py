import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy import stats

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Actuarial Car Risk Model",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Actuarial Car Accident Risk & Pricing Model")
st.write(
    "A frequency–severity model using a **Poisson GLM** for claim frequency and a **Gamma GLM** "
    "for claim severity, combined via the **pure premium** method. Credibility theory (Bühlmann) "
    "is applied to blend individual and population estimates. Adjust inputs to explore your risk profile."
)
st.divider()

# =============================================================
# SECTION 1 — INPUTS
# =============================================================
st.subheader("📋 Risk Factors")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Driver Profile**")
    age = st.slider("Age", 16, 80, 25)
    experience = st.slider("Years of Driving Experience", 0, 60, 5)
    miles = st.slider("Miles Driven per Year", 0, 30000, 10000, step=500)
    driving_type = st.selectbox(
        "Driving Environment",
        ["Highway", "Mixed", "City"],
        help="City driving has ~2× the incident rate of highway per mile (NHTSA)."
    )

with col2:
    st.markdown("**🚙 Vehicle Profile**")
    car_year = st.slider("Car Model Year", 2000, 2025, 2018)
    car_type = st.selectbox(
        "Vehicle Type",
        ["Sedan", "SUV / Truck", "Sports Car", "Motorcycle"],
    )
    has_safety = st.checkbox("Advanced Safety Features (ADAS)", value=True,
                             help="Lane assist, auto-brake, etc. ~15% crash reduction (IIHS).")
    anti_theft = st.checkbox("Anti-Theft / Tracking Device", value=False,
                             help="Reduces theft-related claims.")

with col3:
    st.markdown("**⚠️ Behavioral & Financial Risk**")
    prior_claims = st.selectbox(
        "Prior Claims in Last 3 Years",
        [0, 1, 2, 3],
        help="Actuarial experience rating: past claims predict future ones."
    )
    dui_history = st.checkbox("DUI / DWI in Last 5 Years", value=False,
                              help="DUI roughly doubles accident frequency (NHTSA).")
    credit_tier = st.selectbox(
        "Insurance Credit Score Tier",
        ["Excellent (750+)", "Good (670–749)", "Fair (580–669)", "Poor (<580)"],
        help="Credit score correlates with claim likelihood in most states."
    )
    weather_zone = st.selectbox(
        "Primary Weather Zone",
        ["Mild / Sunny", "Mixed Seasons", "Heavy Snow / Ice", "Heavy Rain / Storms"],
        help="Adverse weather significantly increases collision risk."
    )

st.divider()

# =============================================================
# SECTION 2 — GLM COEFFICIENT TABLES
# These represent log-linear (multiplicative) relativities,
# as used in a real Poisson GLM for frequency.
# Base level (relativity = 1.0) is a typical 35-year-old sedan
# driver with good credit, no claims, mild weather.
# =============================================================

# --- FREQUENCY MODEL (Poisson GLM, log link) ---
# All factors enter as multiplicative relativities on claim rate λ

# Base annual claim rate (per policy year) for the base risk class
BASE_LAMBDA = 0.12   # 12 claims per 100 policy-years — typical for a standard auto book

# Age relativity: U-shaped quadratic on log scale
age_rel = math.exp(0.0018 * (age - 35) ** 2)

# Experience relativity (diminishing returns beyond 15 yrs)
exp_rel = math.exp(-0.055 * min(experience, 15))

# Mileage relativity: Poisson exposure — mileage is the "offset" in a true GLM
# λ scales linearly with exposure (miles), normalized to 10,000 mi/yr = 1.0
miles_rel = miles / 10000

# Driving environment
env_rel_map = {"Highway": 1.00, "Mixed": 1.30, "City": 1.65}
env_rel = env_rel_map[driving_type]

# Vehicle type
vtype_freq_rel_map = {
    "Sedan": 1.00,
    "SUV / Truck": 1.10,
    "Sports Car": 1.55,
    "Motorcycle": 3.10   # ~29× fatality rate, moderate on claim freq
}
vtype_freq_rel = vtype_freq_rel_map[car_type]

# ADAS safety discount
adas_rel = 0.85 if has_safety else 1.00

# Prior claims — experience rating (Bayesian credibility flavor)
prior_claims_rel_map = {0: 1.00, 1: 1.35, 2: 1.75, 3: 2.30}
prior_rel = prior_claims_rel_map[prior_claims]

# DUI surcharge
dui_rel = 2.05 if dui_history else 1.00

# Credit score tier (ISO/industry standard usage-based)
credit_rel_map = {
    "Excellent (750+)": 0.82,
    "Good (670–749)": 1.00,
    "Fair (580–669)": 1.28,
    "Poor (<580)": 1.65
}
credit_rel = credit_rel_map[credit_tier]

# Weather zone
weather_rel_map = {
    "Mild / Sunny": 1.00,
    "Mixed Seasons": 1.20,
    "Heavy Snow / Ice": 1.50,
    "Heavy Rain / Storms": 1.38
}
weather_rel = weather_rel_map[weather_zone]

# Combine all relativities multiplicatively (this IS the Poisson GLM prediction)
lambda_hat = (
    BASE_LAMBDA
    * age_rel
    * exp_rel
    * miles_rel
    * env_rel
    * vtype_freq_rel
    * adas_rel
    * prior_rel
    * dui_rel
    * credit_rel
    * weather_rel
)

# Poisson probability of at least one claim in the policy year
# P(N >= 1) = 1 - P(N = 0) = 1 - e^(-λ)
prob_at_least_one = 1 - math.exp(-lambda_hat)

# Full Poisson PMF for display (P(N=0), P(N=1), P(N=2), P(N=3+))
def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

p0 = poisson_pmf(lambda_hat, 0)
p1 = poisson_pmf(lambda_hat, 1)
p2 = poisson_pmf(lambda_hat, 2)
p3plus = 1 - p0 - p1 - p2

# =============================================================
# SECTION 3 — SEVERITY MODEL (Gamma GLM, log link)
# Gamma distribution is the actuarial standard for claim severity
# because: right-skewed, positive support, constant CV assumption
# =============================================================

# Base severity (mean cost per claim) for base risk class
BASE_SEVERITY = 8_500   # dollars

# Vehicle age factor (newer = more expensive sensors/parts)
car_age_sev = 1 + 0.028 * max(0, car_year - 2010)

# Vehicle type severity multiplier
vtype_sev_rel_map = {
    "Sedan": 1.00,
    "SUV / Truck": 1.22,
    "Sports Car": 1.48,
    "Motorcycle": 2.60   # high medical costs
}
vtype_sev_rel = vtype_sev_rel_map[car_type]

# Weather severity uplift (multi-vehicle collisions, ice damage)
weather_sev_map = {
    "Mild / Sunny": 1.00,
    "Mixed Seasons": 1.10,
    "Heavy Snow / Ice": 1.22,
    "Heavy Rain / Storms": 1.15
}
weather_sev_rel = weather_sev_map[weather_zone]

# DUI severity uplift (high-speed, catastrophic claims)
dui_sev_rel = 1.45 if dui_history else 1.00

# Anti-theft discount on theft/vandalism component (~15% of severity book)
theft_component = 0.15
antitheft_sev_factor = (1 - theft_component * 0.40) if anti_theft else 1.00

# Gamma GLM prediction: multiplicative
mu_severity = (
    BASE_SEVERITY
    * car_age_sev
    * vtype_sev_rel
    * weather_sev_rel
    * dui_sev_rel
    * antitheft_sev_factor
)

# Gamma shape parameter (α) — assumed constant dispersion φ = 1/α
# A typical auto severity dispersion φ ≈ 1.5 → α ≈ 0.67
GAMMA_ALPHA = 0.67
gamma_scale = mu_severity / GAMMA_ALPHA   # scale θ = μ / α

# 80th percentile severity (useful for risk management)
severity_p80 = stats.gamma.ppf(0.80, a=GAMMA_ALPHA, scale=gamma_scale)
severity_p95 = stats.gamma.ppf(0.95, a=GAMMA_ALPHA, scale=gamma_scale)

# =============================================================
# SECTION 4 — PURE PREMIUM & ACTUARIAL OUTPUTS
# Pure Premium = E[N] × E[S]  (frequency × severity)
#             = λ_hat × μ_severity
# This is valid under independence of freq and severity.
# =============================================================

pure_premium = lambda_hat * mu_severity

# Loss ratio target: assume insurer targets 65% loss ratio
# Gross premium = Pure Premium / Loss Ratio Target
LOSS_RATIO_TARGET = 0.65
EXPENSE_RATIO = 0.28      # acquisition + overhead
PROFIT_MARGIN = 0.07      # target underwriting profit

gross_premium = pure_premium / LOSS_RATIO_TARGET

# Indicated premium via expense/profit loading
# Gross Premium = Pure Premium / (1 - expense_ratio - profit_margin)
indicated_premium = pure_premium / (1 - EXPENSE_RATIO - PROFIT_MARGIN)

# =============================================================
# SECTION 5 — BÜHLMANN CREDIBILITY
# Blends individual λ_hat with the population mean (BASE_LAMBDA)
# Z = n / (n + k),  k = v/a (within-group / between-group variance)
# For auto insurance, a common k value is around 3–5.
# =============================================================

CREDIBILITY_K = 4.0   # Bühlmann k parameter for personal auto
policy_years = st.sidebar.number_input(
    "Policy Years on File (for credibility)", min_value=0, max_value=30, value=1,
    help="More policy years → higher credibility weight on your own experience."
)

Z = policy_years / (policy_years + CREDIBILITY_K)   # credibility factor ∈ [0,1)
lambda_credibility = Z * lambda_hat + (1 - Z) * BASE_LAMBDA
pure_premium_credibility = lambda_credibility * mu_severity
prob_credibility = 1 - math.exp(-lambda_credibility)

# =============================================================
# SECTION 6 — RISK TIER (based on credibility-adjusted λ)
# =============================================================

if lambda_credibility < 0.08:
    tier, tier_color = "Preferred", "🟢"
elif lambda_credibility < 0.16:
    tier, tier_color = "Standard", "🟡"
elif lambda_credibility < 0.28:
    tier, tier_color = "Non-Standard", "🟠"
else:
    tier, tier_color = "High Risk / Substandard", "🔴"

# =============================================================
# SECTION 7 — RESULTS DASHBOARD
# =============================================================
st.subheader("📊 Actuarial Risk Dashboard")

# Row 1: Frequency outputs
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
r1c1.metric("Expected Claims / Year (λ)", f"{lambda_hat:.4f}")
r1c2.metric("P(≥1 Claim)", f"{prob_at_least_one:.1%}")
r1c3.metric("Credibility Factor (Z)", f"{Z:.2f}")
r1c4.metric("Credibility-Adj λ", f"{lambda_credibility:.4f}")
r1c5.metric("Risk Class", f"{tier_color} {tier}")

st.write("")

# Row 2: Severity & premium outputs
r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
r2c1.metric("Mean Severity (Gamma μ)", f"${mu_severity:,.0f}")
r2c2.metric("Severity P80", f"${severity_p80:,.0f}")
r2c3.metric("Severity P95", f"${severity_p95:,.0f}")
r2c4.metric("Pure Premium", f"${pure_premium_credibility:,.0f}")
r2c5.metric("Indicated Gross Premium", f"${pure_premium_credibility / (1 - EXPENSE_RATIO - PROFIT_MARGIN):,.0f}")

st.divider()

# =============================================================
# SECTION 8 — POISSON CLAIM COUNT DISTRIBUTION
# =============================================================
st.subheader("📈 Claim Count Distribution (Poisson)")
col_dist, col_rel = st.columns(2)

with col_dist:
    st.markdown("**Probability Mass Function — Annual Claims**")
    pmf_data = {
        "Claims": ["0 Claims", "1 Claim", "2 Claims", "3+ Claims"],
        "Probability": [
            round(poisson_pmf(lambda_hat, 0), 4),
            round(poisson_pmf(lambda_hat, 1), 4),
            round(poisson_pmf(lambda_hat, 2), 4),
            round(1 - poisson_pmf(lambda_hat, 0) - poisson_pmf(lambda_hat, 1) - poisson_pmf(lambda_hat, 2), 4)
        ]
    }
    df_pmf = pd.DataFrame(pmf_data).set_index("Claims")
    st.bar_chart(df_pmf)

with col_rel:
    st.markdown("**Relativity Decomposition — Your λ vs. Base**")
    relativity_rows = [
        ("Base Rate", BASE_LAMBDA, 1.000),
        ("Age Effect", BASE_LAMBDA * age_rel, age_rel),
        ("Experience", BASE_LAMBDA * age_rel * exp_rel, exp_rel),
        ("Mileage", BASE_LAMBDA * age_rel * exp_rel * miles_rel, miles_rel),
        ("Environment", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel, env_rel),
        ("Vehicle Type", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel * vtype_freq_rel, vtype_freq_rel),
        ("Safety/ADAS", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel * vtype_freq_rel * adas_rel, adas_rel),
        ("Prior Claims", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel * vtype_freq_rel * adas_rel * prior_rel, prior_rel),
        ("DUI Surcharge", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel * vtype_freq_rel * adas_rel * prior_rel * dui_rel, dui_rel),
        ("Credit Score", BASE_LAMBDA * age_rel * exp_rel * miles_rel * env_rel * vtype_freq_rel * adas_rel * prior_rel * dui_rel * credit_rel, credit_rel),
        ("Weather Zone", lambda_hat, weather_rel),
    ]
    df_rel = pd.DataFrame(relativity_rows, columns=["Factor", "Cumulative λ", "Relativity"])
    df_rel = df_rel.set_index("Factor")
    st.dataframe(df_rel.style.format({"Cumulative λ": "{:.5f}", "Relativity": "{:.3f}"}), use_container_width=True)

st.divider()

# =============================================================
# SECTION 9 — GAMMA SEVERITY DISTRIBUTION CURVE
# =============================================================
st.subheader("📉 Claim Severity Distribution (Gamma)")

col_gam1, col_gam2 = st.columns(2)

with col_gam1:
    st.markdown("**Gamma PDF — Severity Probability Density**")
    x_range = np.linspace(100, mu_severity * 4, 400)
    pdf_vals = stats.gamma.pdf(x_range, a=GAMMA_ALPHA, scale=gamma_scale)
    df_gamma = pd.DataFrame({"Cost ($)": x_range, "Density": pdf_vals}).set_index("Cost ($)")
    st.line_chart(df_gamma)

with col_gam2:
    st.markdown("**Cumulative Severity (CDF) — % of claims below cost**")
    cdf_vals = stats.gamma.cdf(x_range, a=GAMMA_ALPHA, scale=gamma_scale)
    df_cdf = pd.DataFrame({"Cost ($)": x_range, "Cumulative Probability": cdf_vals}).set_index("Cost ($)")
    st.line_chart(df_cdf)

st.divider()

# =============================================================
# SECTION 10 — SENSITIVITY ANALYSIS
# =============================================================
st.subheader("🔍 Sensitivity Analysis")

sc1, sc2 = st.columns(2)

# Helper: recompute λ from scratch for sensitivity sweeps
def compute_lambda(age_v, exp_v, miles_v, env_v, vtype_v, safety_v,
                   prior_v, dui_v, credit_v, weather_v):
    ar = math.exp(0.0018 * (age_v - 35) ** 2)
    er = math.exp(-0.055 * min(exp_v, 15))
    mr = miles_v / 10000
    env_r = env_rel_map[env_v]
    vt_r = vtype_freq_rel_map[vtype_v]
    saf_r = 0.85 if safety_v else 1.00
    pr_r = prior_claims_rel_map[prior_v]
    dui_r = 2.05 if dui_v else 1.00
    cr_r = credit_rel_map[credit_v]
    wr_r = weather_rel_map[weather_v]
    return BASE_LAMBDA * ar * er * mr * env_r * vt_r * saf_r * pr_r * dui_r * cr_r * wr_r

with sc1:
    st.markdown("**λ vs. Age** (all else equal)")
    ages = np.arange(16, 81)
    lambdas_age = [compute_lambda(a, experience, miles, driving_type, car_type, has_safety,
                                   prior_claims, dui_history, credit_tier, weather_zone)
                   for a in ages]
    df_age = pd.DataFrame({"Age": ages, "Expected Claims λ": lambdas_age}).set_index("Age")
    st.line_chart(df_age)

with sc2:
    st.markdown("**λ vs. Miles Driven**")
    mile_range = np.linspace(500, 30000, 100)
    lambdas_mi = [compute_lambda(age, experience, m, driving_type, car_type, has_safety,
                                  prior_claims, dui_history, credit_tier, weather_zone)
                  for m in mile_range]
    df_mi = pd.DataFrame({"Miles / Year": mile_range, "Expected Claims λ": lambdas_mi}).set_index("Miles / Year")
    st.line_chart(df_mi)

sc3, sc4 = st.columns(2)

with sc3:
    st.markdown("**Pure Premium vs. Credibility (Policy Years)**")
    years_range = np.arange(0, 21)
    pp_range = []
    for y in years_range:
        z = y / (y + CREDIBILITY_K)
        lam_c = z * lambda_hat + (1 - z) * BASE_LAMBDA
        pp_range.append(lam_c * mu_severity)
    df_pp = pd.DataFrame({"Policy Years": years_range, "Pure Premium ($)": pp_range}).set_index("Policy Years")
    st.line_chart(df_pp)
    st.caption("Shows how premium converges to your individual estimate as credibility increases.")

with sc4:
    st.markdown("**Pure Premium by Prior Claims × Credit Tier**")
    prior_list = [0, 1, 2, 3]
    credit_list = ["Excellent (750+)", "Good (670–749)", "Fair (580–669)", "Poor (<580)"]
    matrix = {}
    for ct in credit_list:
        row = []
        for pc in prior_list:
            lam = compute_lambda(age, experience, miles, driving_type, car_type,
                                  has_safety, pc, dui_history, ct, weather_zone)
            row.append(round(lam * mu_severity, 0))
        matrix[ct] = row
    df_matrix = pd.DataFrame(matrix, index=[f"{p} Claims" for p in prior_list])
    df_matrix.index.name = "Prior Claims"
    st.dataframe(df_matrix.style.format("${:,.0f}").background_gradient(cmap="RdYlGn_r", axis=None),
                 use_container_width=True)
    st.caption("Heat map: darker red = higher pure premium.")

st.divider()

# =============================================================
# SECTION 11 — ACTUARIAL OUTPUTS TABLE
# =============================================================
st.subheader("📋 Full Actuarial Summary")

actuarial_rows = {
    "Metric": [
        "Expected Claim Frequency (λ̂)",
        "Credibility Factor (Z)",
        "Credibility-Adjusted λ",
        "P(0 claims this year)",
        "P(1 claim this year)",
        "P(2 claims this year)",
        "P(3+ claims this year)",
        "Mean Severity (Gamma μ)",
        "Severity Std Dev (Gamma σ)",
        "Severity 80th Percentile",
        "Severity 95th Percentile",
        "Pure Premium (unadjusted)",
        "Pure Premium (credibility-adj)",
        "Indicated Gross Premium",
        "Implied Loss Ratio at Market Rate",
    ],
    "Value": [
        f"{lambda_hat:.5f}",
        f"{Z:.3f}",
        f"{lambda_credibility:.5f}",
        f"{poisson_pmf(lambda_hat, 0):.3%}",
        f"{poisson_pmf(lambda_hat, 1):.3%}",
        f"{poisson_pmf(lambda_hat, 2):.3%}",
        f"{p3plus:.3%}",
        f"${mu_severity:,.2f}",
        f"${mu_severity / math.sqrt(GAMMA_ALPHA):,.2f}",
        f"${severity_p80:,.2f}",
        f"${severity_p95:,.2f}",
        f"${pure_premium:,.2f}",
        f"${pure_premium_credibility:,.2f}",
        f"${indicated_premium:,.2f}",
        f"{LOSS_RATIO_TARGET:.0%} (target)",
    ],
    "Actuarial Note": [
        "Poisson GLM — multiplicative relativities",
        "Bühlmann (k=4); converges to 1.0 with 20+ years",
        "Z × λ̂ + (1−Z) × μ_pop",
        "PMF: e^−λ",
        "PMF: λe^−λ",
        "PMF: (λ²/2)e^−λ",
        "1 − P(0) − P(1) − P(2)",
        "Gamma GLM — log link, multiplicative factors",
        "σ = μ / √α  (Gamma property)",
        "stats.gamma.ppf(0.80)",
        "stats.gamma.ppf(0.95)",
        "E[N] × E[S] — independence assumed",
        "Bühlmann-blended λ × E[S]",
        "PP / (1 − expense_ratio − profit_margin)",
        "Industry standard target",
    ]
}

df_actuarial = pd.DataFrame(actuarial_rows)
st.dataframe(df_actuarial, use_container_width=True, hide_index=True)

st.divider()

# =============================================================
# SECTION 12 — MODEL TRANSPARENCY
# =============================================================
with st.expander("⚙️ Statistical Model Documentation"):
    st.markdown("""
    ## Frequency Model — Poisson GLM (Log Link)

    In a Poisson GLM, the expected claim count λ is modeled as:

    > **log(λ) = log(exposure) + β₀ + β₁x₁ + β₂x₂ + … + βₖxₖ**

    Which, after exponentiating, gives the **multiplicative relativity structure**:

    > **λ = exposure × e^β₀ × e^(β₁x₁) × … × e^(βₖxₖ)**

    Each factor's exponentiated coefficient is a **relativity** — how much it scales the base rate.
    The mileage term acts as the **exposure offset** (log(miles/10000)).

    | Factor | Relativity used | Statistical basis |
    |---|---|---|
    | Age | exp(0.0018 × (age−35)²) | Quadratic on log scale; U-shaped (NHTSA) |
    | Experience | exp(−0.055 × min(exp,15)) | Log-linear, capped at 15 yrs |
    | Miles | miles / 10,000 | Exposure offset: λ ∝ exposure |
    | Environment | 1.00 / 1.30 / 1.65 | City ~65% higher than highway |
    | Vehicle type | 1.00 – 3.10 | Motorcycles ~3× higher frequency |
    | ADAS features | 0.85 | ~15% reduction (IIHS 2023) |
    | Prior claims | 1.00 – 2.30 | Experience rating / a posteriori |
    | DUI | 2.05 | ~2× frequency (NHTSA, MADD data) |
    | Credit score | 0.82 – 1.65 | ISO actuarial filing standard |
    | Weather zone | 1.00 – 1.50 | Ice/snow states: ~50% uplift |

    ---

    ## Severity Model — Gamma GLM (Log Link)

    The **Gamma distribution** is the actuarial standard for claim severity because:
    - Support is (0, ∞) — costs are always positive
    - Right skew — a few large losses dominate
    - Constant coefficient of variation (CV = 1/√α) — variance scales with mean²
    - Log-link gives multiplicative structure identical to frequency

    > **E[Severity] = μ = BASE × car_age_factor × vehicle_type × weather × DUI × theft**

    Gamma parameters: **shape α = 0.67**, **scale θ = μ / α**

    This gives a **coefficient of variation CV = 1/√α ≈ 1.22**, implying high spread —
    consistent with auto claim data where catastrophic claims create heavy right tails.

    ---

    ## Pure Premium

    > **Pure Premium = E[N] × E[S] = λ × μ**

    Valid under **independence** of frequency and severity — a standard actuarial assumption
    that holds well in personal auto (unlike commercial lines where large risks correlate freq/sev).

    ---

    ## Bühlmann Credibility

    Blends individual experience with the population mean:

    > **λ_cred = Z × λ̂_individual + (1 − Z) × μ_population**
    
    > **Z = n / (n + k)**,   where **k = v / a** (within-variance / between-variance)

    With k = 4 (typical personal auto), **Z reaches 0.80 at 16 policy years**.
    This prevents over-reacting to a single bad year for a new policyholder.

    ---

    ## Gross Premium Loading

    > **Indicated Premium = Pure Premium / (1 − expense_ratio − profit_margin)**
    >                     = Pure Premium / (1 − 0.28 − 0.07)
    >                     = Pure Premium / 0.65

    This implies a **65% target loss ratio**, consistent with personal auto industry averages.

    ---

    ## Limitations
    - Coefficients are **illustrative** — a real GLM is fitted via MLE on thousands of policy-years
    - No **interaction terms** (e.g., young × sports car is additive here, not multiplicative beyond the individual effects)
    - No **territorial rating** (ZIP-level granularity used in real ratemaking)
    - **Independence assumption** (freq ⊥ severity) may not hold for catastrophic events
    - No **loss development** or IBNR reserves modeled
    - Credit score use is **banned in some states** (CA, MA, HI)
    """)

with st.expander("📚 References"):
    st.markdown("""
    - **Dobson & Barnett (2018)** — *An Introduction to Generalized Linear Models* (3rd ed.)
    - **Klugman, Panjer & Willmot** — *Loss Models: From Data to Decisions* (Wiley, 4th ed.)
    - **Mack (1994)** — Credibility for sums of independent Poisson variables
    - **Bühlmann (1967)** — "Experience rating and credibility" — *ASTIN Bulletin*
    - **IIHS (2023)** — Crash avoidance features effectiveness study
    - **NHTSA FARS database** — Fatality Analysis Reporting System
    - **ISO CGL filing** — Credit score actuarial filing documentation
    """)
