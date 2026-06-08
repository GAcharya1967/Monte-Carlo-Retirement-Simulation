import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =============================
# Page Config
# =============================
st.set_page_config(layout="wide")
st.sidebar.title("Monte Carlo Inputs")

# =============================
# Helper: Money formatter
# =============================
def format_money(x):
    if x >= 1e7:
        return f"{x/1e7:.1f} Cr"
    elif x >= 1e5:
        return f"{x/1e5:.0f} L"
    else:
        return f"₹{x:,.0f}"

# =============================
# Core Inputs
# =============================
st.sidebar.subheader("Core Inputs")

colA, colB = st.sidebar.columns(2)

with colA:
    start_corpus = st.number_input(
        "Total Corpus (₹)",
        value=100_000_000,
        min_value=0,
        step=5_000_000
    )
    st.caption(f"→ {format_money(start_corpus)}")

with colB:
    expected_return = st.slider(
        "Expected Return (%)",
        4.0, 15.0, 10.0, 0.25
    ) / 100.0

colC, colD = st.sidebar.columns(2)

with colC:
    volatility = st.slider(
        "Return Volatility (%)",
        5.0, 30.0, 15.0, 0.5
    ) / 100.0

with colD:
    inflation = st.slider(
        "Inflation (%)",
        3.0, 10.0, 7.5, 0.25
    ) / 100.0

# =============================
# Asset Allocation
# =============================
st.sidebar.subheader("Asset Allocation")

equity_pct = st.sidebar.slider(
    "Equity Allocation (%)",
    0, 100, 60, 5
)

debt_pct = 100 - equity_pct
DEBT_RETURN = 0.04

st.sidebar.caption(
    f"→ Equity: {equity_pct}% | Debt: {debt_pct}% (assumed 4% return)"
)

# =============================
# Spending
# =============================
st.sidebar.subheader("Spending")

colE, colF = st.sidebar.columns(2)

with colE:
    essential_spend = st.number_input(
        "Essential Spend (₹)",
        value=3_000_000,
        min_value=0,
        step=250_000
    )
    st.caption(f"→ {format_money(essential_spend)}")

with colF:
    discretionary_spend = st.number_input(
        "Discretionary Spend (₹)",
        value=2_000_000,
        min_value=0,
        step=250_000
    )
    st.caption(f"→ {format_money(discretionary_spend)}")

# =============================
# Years
# =============================
years = st.sidebar.slider("Years of Retirement", 10, 60, 40)

# =============================
# Risk Protection
# =============================
with st.sidebar.expander("🛡️ Risk Protection", expanded=False):
    safe_years = st.slider("Safe Bucket (Years)", 0, 10, 5)
    cut_pct = st.slider("Discretionary Cut in Bad Years (%)", 0, 60, 30) / 100.0

# =============================
# Aging & Longevity
# =============================
with st.sidebar.expander("🧓 Aging & Longevity", expanded=False):
    current_age = st.number_input("Current Age", value=60, min_value=1, max_value=100)

    colG, colH = st.columns(2)
    with colG:
        taper_start_age = st.slider("Spending Taper (Age)", 80, 90, 80)
    with colH:
        taper_pct = st.slider("Taper (%)", 0, 5, 2) / 100.0

# =============================
# Sequence Risk
# =============================
st.sidebar.subheader("Sequence Risk")

force_sequence_risk = st.sidebar.toggle(
    "Force 5-Year Sequence Risk",
    value=False
)

SEQUENCE_YEARS = 5

# =============================
# Simulation Settings
# =============================
SIMS = 50_000

# =============================
# Monte Carlo Engine
# =============================
def run_monte_carlo():
    data = np.zeros((years + 1, SIMS))

    equity_weight = equity_pct / 100.0
    debt_weight = 1.0 - equity_weight
    debt_return = DEBT_RETURN

    for sim in range(SIMS):
        total_spend = essential_spend + discretionary_spend
        safe_bucket = safe_years * total_spend
        risky = max(0, start_corpus - safe_bucket)

        ess = essential_spend
        disc = discretionary_spend
        age = current_age

        data[0, sim] = risky + safe_bucket

        for yr in range(1, years + 1):

            # ---- Equity return ----
            if equity_weight == 0:
                equity_r = 0.0
            else:
                if force_sequence_risk and yr <= SEQUENCE_YEARS:
                    equity_r = np.random.normal(-0.10, 0.05)
                    equity_r = min(equity_r, 0.0)
                else:
                    equity_r = np.random.normal(expected_return, volatility)

            # ---- Portfolio return ----
            r = equity_weight * equity_r + debt_weight * debt_return

            # Apply return FIRST (Excel-consistent)
            risky *= (1 + r)

            # Inflate spending
            ess *= (1 + inflation)
            disc *= (1 + inflation)

            # Taper spending
            if age >= taper_start_age:
                ess *= (1 - taper_pct)
                disc *= (1 - taper_pct)

            # Discretionary cut
            disc_adj = disc * (1 - cut_pct) if r < 0 else disc
            total_draw = ess + disc_adj

            # Withdrawals
            if yr <= safe_years and safe_bucket > 0:
                safe_bucket *= (1 + debt_return)
                draw = min(safe_bucket, total_draw)
                safe_bucket -= draw
                risky -= (total_draw - draw)
            else:
                risky -= total_draw

            risky = max(0, risky)
            data[yr, sim] = risky + safe_bucket

            if risky == 0 and safe_bucket == 0:
                data[yr + 1 :, sim] = 0
                break

            age += 1

    return pd.DataFrame(data)

# =============================
# Run Simulation
# =============================
df = run_monte_carlo()

# =============================
# Statistics (FIXED AXIS)
# =============================
p10 = df.quantile(0.10, axis=1)
p50 = df.quantile(0.50, axis=1)
p90 = df.quantile(0.90, axis=1)

# Success = never hits zero
success_rate = (df.min(axis=0) > 0).mean()

# =============================
# Median PV
# =============================
terminal_median = np.median(df.iloc[-1].clip(lower=0))
pv_terminal_median = 0 if terminal_median <= 0 else terminal_median / ((1 + inflation) ** years)

# =============================
# Header Boxes
# =============================
col_succ, col_pv = st.columns([1, 1])

succ_bg, succ_fg = (
    ("#22c55e", "black") if success_rate >= 0.75 else
    ("#facc15", "black") if success_rate >= 0.50 else
    ("#ef4444", "white")
)

pv_bg, pv_fg = ("#22c55e", "black") if pv_terminal_median > 0 else ("#ef4444", "white")

with col_succ:
    st.markdown(
        f"""<div style="background:{succ_bg};color:{succ_fg};
        padding:10px 18px;border-radius:8px;font-size:20px;font-weight:600;width:fit-content;">
        Success {success_rate*100:.1f}%</div>""",
        unsafe_allow_html=True
    )

with col_pv:
    st.markdown(
        f"""<div style="float:right;background:{pv_bg};color:{pv_fg};
        padding:10px 18px;border-radius:8px;font-size:18px;font-weight:600;">
        Median PV {format_money(pv_terminal_median)}</div>""",
        unsafe_allow_html=True
    )

# =============================
# Earliest Ruin Path
# =============================
# Find which simulation hits zero earliest
def first_ruin_year(series):
    zeros = np.where(series.values <= 0)[0]
    return zeros[0] if len(zeros) > 0 else len(series)

ruin_by_sim = df.apply(first_ruin_year, axis=0)
earliest_ruin_sim = ruin_by_sim.idxmin()
earliest_ruin_yr = ruin_by_sim.min()
has_ruin = earliest_ruin_yr < years  # at least one sim actually hit zero

# =============================
# Plot
# =============================
fig, ax = plt.subplots(figsize=(10, 6))

for col in df.columns[:200]:
    # Skip the worst path — it gets its own highlight below
    if col == earliest_ruin_sim:
        continue
    ax.plot(df[col], color="#7ec8e3", alpha=0.18, linewidth=0.6)

ax.fill_between(range(len(p10)), p10, p90, alpha=0.25, label="10–90 percentile")
ax.plot(p50, linewidth=3, label="Median")
ax.scatter(0, start_corpus, color="red", s=80, label="Start")

# ---- Highlight the single earliest-ruin path ----
if has_ruin:
    worst_path = df[earliest_ruin_sim]
    ruin_age = int(current_age) + earliest_ruin_yr
    ax.plot(
        worst_path,
        color="#ff6600",
        linewidth=2.0,
        zorder=5,
        label=f"Earliest Ruin (Age {ruin_age})"
    )
    ax.scatter(
        earliest_ruin_yr,
        0,
        color="#ff6600",
        s=100,
        zorder=6,
        marker="X"
    )
    ax.axvline(x=earliest_ruin_yr, color="#ff6600", linewidth=1.2, linestyle="--", alpha=0.7)

ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1e7)} Cr"))

# Cap Y axis at P75 at the midpoint of retirement — prevents long-horizon compounding
# from collapsing the early/mid years into a flat line
mid_yr = min(20, years)
y_cap = max(start_corpus * 3, df.iloc[mid_yr].quantile(0.75) * 2)
ax.set_ylim(0, y_cap)

# X-axis: show age instead of years
x_ticks = range(0, years + 1, 5)
ax.set_xticks(list(x_ticks))
ax.set_xticklabels([int(current_age) + yr for yr in x_ticks])
ax.set_xlabel("Age")

seq_label = "ON" if force_sequence_risk else "OFF"

if equity_pct == 0:
    return_label = "Return 4.0% (Debt)"
else:
    return_label = f"Return {expected_return*100:.1f}%"

ax.set_title(
    f"Corpus {format_money(start_corpus)} | "
    f"Ess {format_money(essential_spend)} + Disc {format_money(discretionary_spend)} | "
    f"{return_label} | Infl {inflation*100:.1f}% | "
    f"Seq Risk {seq_label}"
)

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
