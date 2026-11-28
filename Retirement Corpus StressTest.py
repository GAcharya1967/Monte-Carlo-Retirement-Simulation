"""
Retirement Corpus StressTest (Streamlit)
========================================
Two tools in one app, chosen by the switch at the top of the sidebar:
  Monte Carlo  : probabilistic retirement simulator (random returns and inflation, Guyton-Klinger guardrails).
  Historical Stress Test : replays historical USA, India and Japan return sequences.

Run:   pip install streamlit pandas altair numpy matplotlib
       streamlit run "Retirement Corpus StressTest.py"

--- Historical Stress Test notes ---
Replays historical USA, India and Japan return sequences to test retirement sustainability.

Data (embedded, nothing to download):
  US     : Damodaran histretSP.xls (updated 1 Jan 2026): S&P 500 total return, CPI-U (matches FRED).
  India  : Sensex financial-year (April-March) returns from the BSE "Sensex at 40" paper. FY1986-87 to FY1996-97
           use price return plus an ESTIMATED dividend uplift (the paper's TRI starts at March 1997); FY1997-98 to
           FY2024-25 use the published TRI; FY2025-26 is user-supplied (not in the paper).
           Inflation: World Bank calendar-year CPI converted to April-March years (75% prior year + 25% current year); India keys use the FY April start year. Monthly Combined CPI is a separate validation series from 2011 onward, not a replacement.
  Japan  : TOPIX year-end price return (supplied workbook) plus dividend return (JST Release 6 eq_div_rtn to 2020,
           2.3% estimate for 2021-2025); Japan CPI and cash yields from the supplied workbook. Not official TOPIX TR.
  Debt   : USA = Damodaran 3-month T-bill rate (annual average, same as FRED), unchanged.
           India = selected real-return assumption relative to each year's CPI: match inflation, inflation - 1%,
           or inflation - 2%. Withdrawal tax is a separate user-selected assumption.

--- Monte Carlo notes ---
Retirement Monte Carlo Calculator (v2, personal-use edition)
============================================================
Success means every scheduled withdrawal is fully funded through the selected horizon.
This is a simplified planning model, not financial advice. Taxes and fees are NOT modelled.

Model conventions:
- Age is age at the start of the first retirement year.
- Annual returns are applied first; rebalancing (if due) follows; then the year's
  withdrawal is paid (end-of-year withdrawal).
- Spending inputs are first-retirement-year amounts; inflation begins in year 2.
- Annual taper is a reduction in real spending (essential/discretionary only), applied
  from the selected age onward. Medical spending is never tapered or cut.
- Medical: if a separate medical corpus is planned, medical spending is excluded here.
  Otherwise it is modelled as its own line that inflates at (general inflation + premium).
- Safe bucket (debt-like reserve), two modes:
    * Refill mode (default): drawn only in "down years" (invested-portfolio return < 0),
      refilled from invested assets after non-down years up to
      safe_years x that year's scheduled spending.
    * Original mode: spent first every year, never refilled.
- Equity returns: Student-t (fat tails, 5 df) or normal, scaled to the same mean and
  volatility, truncated at -100%. Debt return is deterministic.
- Reported balances are in today's rupees (deflated by each path's own inflation).
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, replace, asdict
from typing import Optional

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================
# STRESS TEST: data and engine
# ======================================================================
DIV_UPLIFT = 0.01574

# year: [S&P 500 total return, CPI-U inflation, 3-month T-bill rate (annual average, Damodaran/FRED)]
US = {
    1928: [0.43811, -0.01156, 0.0308],
    1929: [-0.08298, 0.00585, 0.0316],
    1930: [-0.25124, -0.06395, 0.0455],
    1931: [-0.43838, -0.09317, 0.0231],
    1932: [-0.08642, -0.10274, 0.0107],
    1933: [0.49982, 0.00763, 0.0096],
    1934: [-0.01189, 0.01515, 0.00278],
    1935: [0.4674, 0.02985, 0.00168],
    1936: [0.31943, 0.01449, 0.00172],
    1937: [-0.35337, 0.02857, 0.00276],
    1938: [0.29283, -0.02778, 0.00065],
    1939: [-0.01098, 0.0, 0.00046],
    1940: [-0.10673, 0.00714, 0.00036],
    1941: [-0.12771, 0.09929, 0.00129],
    1942: [0.19174, 0.09032, 0.00343],
    1943: [0.25061, 0.02959, 0.0038],
    1944: [0.19031, 0.02299, 0.0038],
    1945: [0.35821, 0.02247, 0.0038],
    1946: [-0.08429, 0.18132, 0.0038],
    1947: [0.052, 0.08837, 0.00601],
    1948: [0.05705, 0.02991, 0.01045],
    1949: [0.18303, -0.02075, 0.01115],
    1950: [0.30806, 0.05932, 0.01203],
    1951: [0.23678, 0.06, 0.01517],
    1952: [0.18151, 0.00755, 0.01723],
    1953: [-0.01208, 0.00749, 0.01891],
    1954: [0.52563, -0.00743, 0.00938],
    1955: [0.32597, 0.00375, 0.01724],
    1956: [0.0744, 0.02985, 0.02621],
    1957: [-0.10457, 0.02899, 0.03225],
    1958: [0.4372, 0.01761, 0.01767],
    1959: [0.12056, 0.0173, 0.03386],
    1960: [0.00337, 0.01361, 0.02873],
    1961: [0.26638, 0.00671, 0.02352],
    1962: [-0.08811, 0.01333, 0.02772],
    1963: [0.22612, 0.01645, 0.03156],
    1964: [0.16415, 0.00971, 0.03546],
    1965: [0.12399, 0.01923, 0.03949],
    1966: [-0.09971, 0.03459, 0.04856],
    1967: [0.23803, 0.0304, 0.04293],
    1968: [0.10815, 0.0472, 0.05338],
    1969: [-0.08241, 0.06197, 0.06668],
    1970: [0.03561, 0.0557, 0.06391],
    1971: [0.14221, 0.03266, 0.04334],
    1972: [0.18755, 0.03406, 0.04062],
    1973: [-0.14308, 0.08706, 0.07035],
    1974: [-0.25902, 0.12338, 0.07846],
    1975: [0.36995, 0.06936, 0.05786],
    1976: [0.23831, 0.04865, 0.04977],
    1977: [-0.0698, 0.06701, 0.05261],
    1978: [0.06509, 0.09018, 0.07178],
    1979: [0.18519, 0.13294, 0.10054],
    1980: [0.31735, 0.12516, 0.11392],
    1981: [-0.04702, 0.08922, 0.14036],
    1982: [0.20419, 0.0383, 0.1109],
    1983: [0.22337, 0.03791, 0.0895],
    1984: [0.06146, 0.03949, 0.0992],
    1985: [0.31235, 0.03799, 0.0772],
    1986: [0.18495, 0.01098, 0.0615],
    1987: [0.05813, 0.04434, 0.0596],
    1988: [0.16537, 0.04419, 0.0689],
    1989: [0.31475, 0.04647, 0.0839],
    1990: [-0.03064, 0.06106, 0.0775],
    1991: [0.30235, 0.03064, 0.0554],
    1992: [0.07494, 0.02901, 0.0351],
    1993: [0.09967, 0.02748, 0.0307],
    1994: [0.01326, 0.02675, 0.0437],
    1995: [0.37195, 0.02538, 0.0566],
    1996: [0.22681, 0.03322, 0.0515],
    1997: [0.33104, 0.01702, 0.052],
    1998: [0.28338, 0.01612, 0.0491],
    1999: [0.20885, 0.02685, 0.0478],
    2000: [-0.09032, 0.03387, 0.06],
    2001: [-0.1185, 0.01552, 0.0348],
    2002: [-0.21966, 0.02377, 0.0164],
    2003: [0.28356, 0.01879, 0.0103],
    2004: [0.10743, 0.03256, 0.014],
    2005: [0.04834, 0.03416, 0.0322],
    2006: [0.15613, 0.02541, 0.0485],
    2007: [0.05485, 0.04081, 0.0448],
    2008: [-0.36552, 0.00091, 0.014],
    2009: [0.25935, 0.02721, 0.0015],
    2010: [0.14821, 0.01496, 0.0014],
    2011: [0.02098, 0.02962, 0.0005],
    2012: [0.15891, 0.01741, 0.0009],
    2013: [0.32145, 0.01502, 0.0006],
    2014: [0.13524, 0.00756, 0.0003],
    2015: [0.01379, 0.0073, 0.0005],
    2016: [0.11773, 0.02075, 0.0032],
    2017: [0.21605, 0.02109, 0.0095],
    2018: [-0.04227, 0.0191, 0.0197],
    2019: [0.31212, 0.02285, 0.0211],
    2020: [0.18023, 0.01362, 0.0036],
    2021: [0.28469, 0.07036, 0.0004],
    2022: [-0.18038, 0.06454, 0.0209],
    2023: [0.26061, 0.03352, 0.0528],
    2024: [0.24879, 0.02888, 0.0518],
    2025: [0.17724, 0.02735, 0.0421],
}

# key = FY start year (1986 = FY1986-87, April-March).
# [Sensex total return, FY CPI inflation, 1 if dividend component estimated, debt return]
# Debt: FY1986-87 to FY1992-93 = RBI weighted-average yield on new central-government bonds (no T-bill market existed;
# the 91-day tap bill was fixed at about 4.6%). FY1993-94 onward = simple average of the weekly 91-day T-bill auction cut-off yields (RBI).
# FY CPI = 75% of the prior calendar year + 25% of the calendar year of the March year-end (World Bank CPI).
# FY2025-26 reuses the 2025 rate because 2026 is not yet published.
INDIA = {
    1986: [-0.09526, 0.08748, 1, 0.11380],
    1987: [-0.20366, 0.08947, 1, 0.11250],
    1988: [0.80704, 0.08806, 1, 0.11400],
    1989: [0.11024, 0.07549, 1, 0.11490],
    1990: [0.51114, 0.10196, 1, 0.11410],
    1991: [2.68454, 0.13350, 1, 0.11780],
    1992: [-0.45206, 0.10423, 1, 0.12460],
    1993: [0.67284, 0.07307, 1, 0.08871],
    1994: [-0.12136, 0.10242, 1, 0.09163],
    1995: [0.04814, 0.09913, 1, 0.12671],
    1996: [0.01404, 0.08524, 1, 0.09666],
    1997: [0.17840, 0.08681, 0, 0.06832],
    1998: [-0.01990, 0.11091, 0, 0.08561],
    1999: [0.37380, 0.04505, 0, 0.09033],
    2000: [-0.27140, 0.03952, 0, 0.08968],
    2001: [-0.01200, 0.03909, 0, 0.06949],
    2002: [-0.10380, 0.04174, 0, 0.05810],
    2003: [0.87890, 0.03796, 0, 0.04586],
    2004: [0.18650, 0.03887, 0, 0.04927],
    2005: [0.76670, 0.04634, 0, 0.05693],
    2006: [0.17790, 0.05941, 0, 0.06632],
    2007: [0.20830, 0.06867, 0, 0.07115],
    2008: [-0.37080, 0.08983, 0, 0.07101],
    2009: [0.82790, 0.11159, 0, 0.03566],
    2010: [0.12530, 0.11220, 0, 0.06176],
    2011: [-0.09160, 0.09054, 0, 0.08429],
    2012: [0.10130, 0.09614, 0, 0.08199],
    2013: [0.20730, 0.09180, 0, 0.08901],
    2014: [0.26810, 0.06226, 0, 0.08499],
    2015: [-0.07910, 0.04917, 0, 0.07427],
    2016: [0.18460, 0.04543, 0, 0.06417],
    2017: [0.12700, 0.03481, 0, 0.06194],
    2018: [0.18770, 0.03886, 0, 0.06610],
    2019: [-0.22860, 0.04453, 0, 0.05454],
    2020: [0.69820, 0.06250, 0, 0.03289],
    2021: [0.19500, 0.05523, 0, 0.03509],
    2022: [0.02030, 0.06437, 0, 0.05793],
    2023: [0.26500, 0.05475, 0, 0.06872],
    2024: [0.06390, 0.04314, 0, 0.06641],
    2025: [-0.06010, 0.02399, 0, 0.05493],
}


# Japan: annual calendar-year returns, CPI and cash yields for 1989-2025.
# Equity return = TOPIX price return (year-end to year-end, supplied Japan workbook) + dividend return.
#   1989-2020: dividend return = JST Macrohistory Database Release 6 eq_div_rtn (an independent estimate of dividends paid / prior price).
#   2021-2025: dividend return is an ESTIMATE, JAPAN_DIV_EST (about the average JST dividend return of 2018-2020).
# The JST eq_tr series itself is NOT used: it does not track TOPIX year by year (e.g. 1990: -13% vs TOPIX -40%).
# CPI = World Bank via FRED; cash = FRED annual-average Japan T-bill yield (both from the supplied workbook).
JAPAN_DIV_EST = 0.023
# year: [TOPIX price return, dividend return, CPI inflation, cash yield]
JAPAN_SRC = {
    1989: [0.222458, 0.005383, 0.0227, 0.0460],
    1990: [-0.398262, 0.004189, 0.0308, 0.0710],
    1991: [-0.011045, 0.005169, 0.0325, 0.0660],
    1992: [-0.237374, 0.006423, 0.0176, 0.0420],
    1993: [0.100676, 0.008825, 0.0124, 0.0240],
    1994: [0.083220, 0.007459, 0.0070, 0.0220],
    1995: [0.011936, 0.007246, -0.0013, 0.0110],
    1996: [-0.067668, 0.008348, 0.0014, 0.0035],
    1997: [-0.201171, 0.007963, 0.0176, 0.0034],
    1998: [-0.074926, 0.009727, 0.0066, 0.0033],
    1999: [0.584375, 0.011811, -0.0034, 0.0008],
    2000: [-0.254634, 0.010151, -0.0068, 0.0012],
    2001: [-0.195946, 0.008881, -0.0074, 0.0006],
    2002: [-0.182969, 0.010689, -0.0092, 0.0001],
    2003: [0.237641, 0.012078, -0.0026, 0.0001],
    2004: [0.101505, 0.013740, -0.0001, 0.0001],
    2005: [0.435036, 0.012386, -0.0028, 0.0001],
    2006: [0.018979, 0.013246, 0.0025, 0.0010],
    2007: [-0.122178, 0.012608, 0.0006, 0.0050],
    2008: [-0.417733, 0.013787, 0.0138, 0.0047],
    2009: [0.056271, 0.016476, -0.0135, 0.0015],
    2010: [-0.009685, 0.019495, -0.0072, 0.0010],
    2011: [-0.189352, 0.018967, -0.0027, 0.0010],
    2012: [0.180055, 0.020418, -0.0006, 0.0010],
    2013: [0.514643, 0.027994, 0.0034, 0.0009],
    2014: [0.080796, 0.017300, 0.0276, 0.0007],
    2015: [0.099317, 0.016233, 0.0080, 0.0003],
    2016: [-0.018542, 0.021700, -0.0012, -0.0015],
    2017: [0.196858, 0.025400, 0.0048, -0.0010],
    2018: [-0.177969, 0.018200, 0.0099, -0.0012],
    2019: [0.152113, 0.028800, 0.0047, -0.0014],
    2020: [0.048404, 0.024000, -0.0002, -0.0010],
    2021: [0.103980, JAPAN_DIV_EST, -0.0023, -0.0010],
    2022: [-0.050504, JAPAN_DIV_EST, 0.0250, -0.0010],
    2023: [0.250926, JAPAN_DIV_EST, 0.0327, -0.0005],
    2024: [0.176864, JAPAN_DIV_EST, 0.0274, 0.0010],
    2025: [0.224082, JAPAN_DIV_EST, 0.0317, 0.0050],
}
# Combined series used by the model: [equity total return, CPI inflation, cash yield]
JAPAN = {y: [v[0] + v[1], v[2], v[3]] for y, v in JAPAN_SRC.items()}

SCENARIOS = {
    "USA": [(1929, 50), (1966, 50)],
    "India": [(1986, 40), (2000, 26)],
    "Japan": [(1990, 36), (2000, 26)],
}
# Amounts are held in rupees crore. Sliders: corpus 0.25-50 Cr, expenses 1-150 Lakh.
CORPUS_OPTIONS = [round(0.25 * k, 2) for k in range(1, 201)]   # Cr
SPEND_OPTIONS = list(range(1, 151))                              # Lakh
SUCCESS_BAR = 85
START_AGE = 60  # retirement is assumed to start at age 60
PALETTE = ["#2563eb", "#f97316"]


def rates(country, year, india_debt_real_assumption=0.0):
    """Return annual equity return, debt return and inflation. India years are FY start-year labels.
    India debt compounds that year's CPI with the selected real-return assumption."""
    if country == "USA":
        eq, cpi, debt = US[year]
    elif country == "India":
        eq, cpi, _, _historical_debt = INDIA[year]
        debt = (1 + cpi) * (1 + india_debt_real_assumption) - 1
    else:
        eq, cpi, debt = JAPAN[year]
    return eq, debt, cpi



def ylab(country, y):
    """Label a year. India key 2019 is the financial year that starts April 2019 (FY2019-20); it is shown as 2019."""
    return str(y)


def scenario_context(country, year):
    """Concise historical context for each selected retirement start year."""
    context = {
        "USA": {
            1929: "Great Depression",
            1966: "Inflation and rising interest rates",
        },
        "India": {
            1986: "High inflation and weak market returns",
            2000: "Dot-com bust and sharp market decline",
        },
        "Japan": {
            1990: "Asset-price bubble collapse",
            2000: "Dot-com downturn and deflation",
        },
    }
    return context.get(country, {}).get(year, "")


def slab(country, y):
    """Return the start-year label; India keys already use the FY April start year."""
    return str(y)


def money(v, country=None):
    """v is in crore. Under 1 Cr it is shown in Lakh (1 Cr = 100 Lakh)."""
    if abs(v) >= 1:
        return f"₹{v:,.2f} Cr"
    return f"₹{v * 100:,.1f} L"


USD_INR = 93  # rupees per US dollar: the ONE line to edit. Used by the sliders and the chart tooltip when USA is selected


def usd(v_cr):
    """v_cr is in rupee crore (1 Cr = 10 million rupees); converted at USD_INR and shown in brackets."""
    d = v_cr * 1e7 / USD_INR
    return f"(${d / 1e6:,.2f} M)" if abs(d) >= 1e6 else f"(${d / 1e3:,.0f} K)"


# USA sliders move in round dollar steps: corpus in $0.1 M steps, expenses in $5 K steps (stored in $ M / $ K).
USD_CORPUS_OPTIONS = [round(0.1 * k, 1) for k in range(1, 55)]    # $ million (0.1 to 5.4)
USD_SPEND_OPTIONS = list(range(5, 165, 5))                        # $ thousand (5 to 160)


def nearest(options, cr_of, cr):
    """Option whose rupee value (crore) is closest to cr."""
    return min(options, key=lambda o: abs(cr_of(o) - cr))


def simulate(country, start, years, corpus, spend, taper, age0, equity, mode, india_debt_real_assumption=0.0, tax_rate=0.10):
    """Annual replay. Withdraw at start of year, then the remainder earns the blended return.
    mode = 'fixed' : inflation-adjusted withdrawals.
    mode = 'gk'    : Guyton-Klinger (skip inflation raise after a down year if withdrawal rate is above
                     initial; cut 10% if rate > 120% of initial, not in last 15 yrs; raise 10% if < 80%).
    Taper: total spending falls by `taper` per year for every year of age above 80, in both modes."""
    bal, cum_infl, w = corpus, 1.0, spend
    prev_ret, prev_t, iwr = 0.0, 1.0, spend / corpus
    rows, ok, fail_year, cuts, skips, events = [], True, None, 0, 0, []
    debt_share = 1 - equity
    for i in range(years):
        y, age = start + i, age0 + i
        t = (1 - taper) ** max(0, age - 80)
        if i > 0:
            infl_prev = rates(country, y - 1, india_debt_real_assumption)[2]
            cum_infl *= 1 + infl_prev
            if mode == "fixed":
                w = spend * cum_infl * t
            else:
                factor, step = 1 + infl_prev, t / prev_t
                if prev_ret < 0 and (w * factor * step) / bal > iwr:
                    if infl_prev < 0:
                        rule_label = (
                            f"Deflation adjustment skipped ({infl_prev:.3%}); spending held at "
                            f"{money(w * step, country)} instead of falling to "
                            f"{money(w * factor * step, country)}"
                        )
                    else:
                        rule_label = "Inflation adjustment skipped (negative portfolio year)"
                    events.append((y, rule_label, w * factor * step, w * step))
                    factor = 1.0
                    skips += 1
                w = w * factor * step
                wr = w / bal
                if i < years - 15 and wr > iwr * 1.2:
                    events.append((y, "10% cut (above upper guardrail)", w, w * 0.9))
                    w *= 0.9
                    cuts += 1
                elif wr < iwr * 0.8:
                    events.append((y, "10% raise (below lower guardrail)", w, w * 1.1))
                    w *= 1.1
        eq, debt, infl = rates(country, y, india_debt_real_assumption)
        ret = equity * eq + (1 - equity) * debt
        tax = w * tax_rate
        total_debit = w + tax
        if bal < total_debit:
            # Record the partial net withdrawal the remaining corpus can fund, including its tax.
            actual_withdrawal = bal / (1 + tax_rate) if (1 + tax_rate) else 0.0
            actual_tax = bal - actual_withdrawal
            rows.append(dict(Year=y, Age=age, Withdrawn=actual_withdrawal, Tax=actual_tax,
                             TotalDebit=bal, MarketReturn=eq, DebtReturn=debt, Return=ret,
                             Inflation=infl, EndBalance=0.0, Real=0.0, Plan=spend * t, CumInfl=cum_infl))
            ok, fail_year = False, y
            break
        bal = (bal - total_debit) * (1 + ret)
        real = bal / (cum_infl * (1 + infl))  # in start-year money
        rows.append(dict(Year=y, Age=age, Withdrawn=w, Tax=tax, TotalDebit=total_debit,
                         MarketReturn=eq, DebtReturn=debt, Return=ret,
                         Inflation=infl, EndBalance=bal, Real=real, Plan=spend * t, CumInfl=cum_infl))
        prev_ret, prev_t = ret, t
    df = pd.DataFrame(rows)
    funded = df if ok else df.iloc[:-1]
    ratio = (funded.Withdrawn / funded.CumInfl) / funded.Plan if len(funded) else pd.Series([0.0])
    if len(funded):
        spent = funded.Withdrawn / funded.CumInfl  # start-year money
        j = (spent / funded.Plan).idxmin()
        avg_spend, avg_plan = float(spent.mean()), float(funded.Plan.mean())
        min_spend, min_plan = float(spent.loc[j]), float(funded.Plan.loc[j])
    else:
        avg_spend = avg_plan = min_spend = min_plan = 0.0
    return dict(start=start, years=years, ok=ok, fail_year=fail_year, df=df,
                avg_spend=avg_spend, avg_plan=avg_plan, min_spend=min_spend, min_plan=min_plan,
                years_funded=years if ok else len(df) - 1,
                end_nominal=bal if ok else 0.0, end_real=float(df.Real.iloc[-1]) if ok else 0.0,
                avg_ratio=float(ratio.mean()), min_ratio=float(ratio.min()), cuts=cuts, skips=skips, events=events)


def rolling(country, years, **kw):
    # Use only complete historical windows. India keys are FY April start years;
    # the first available observation is 1986 (FY1986-87).
    data = US if country == "USA" else INDIA if country == "India" else JAPAN
    available = sorted(data)
    first, last = available[0], available[-1]
    valid_starts = [
        start for start in available
        if all(year in data for year in range(start, start + years))
    ]
    res = [simulate(country, y, years, mode="fixed", **kw)["ok"] for y in valid_starts]
    return sum(res), len(res)


def draw_chart(results, corpus, country):
    recs = []
    for r in results:
        context = scenario_context(country, r["start"])
        name = f"{slab(country, r['start'])} Start" + (f" ({context})" if context else "")
        recs.append(dict(Path=name, Age=START_AGE, Calendar=ylab(country, r["start"]), ValueCr=float(corpus), ValueTxt=money(float(corpus)), ValueUSD=usd(float(corpus))))
        for k, v in enumerate(r["df"].Real, start=1):
            recs.append(dict(Path=name, Age=START_AGE + k, Calendar=ylab(country, r["start"] + k), ValueCr=float(v), ValueTxt=money(float(v)), ValueUSD=usd(float(v))))
        end_real = float(r["end_real"])
        end_nominal = float(r["end_nominal"])
        # Show endpoint labels only for paths with a positive remaining corpus.
        # Zero-balance paths otherwise produce overlapping labels on the baseline.
        if end_real > 0:
            end_label = f"Nominal: {money(end_nominal)} (PV {money(end_real)})"
            recs[-1]["EndLabel"] = end_label
            recs[-1]["EndColor"] = "#16a34a"
    df = pd.DataFrame(recs)
    names = [
        f"{slab(country, r['start'])} Start"
        + (f" ({scenario_context(country, r['start'])})" if scenario_context(country, r["start"]) else "")
        for r in results
    ]
    color = alt.Color("Path:N", scale=alt.Scale(domain=names, range=PALETTE[:len(names)]),
                      legend=None)
    x = alt.X("Age:Q", title="Age", scale=alt.Scale(zero=False), axis=alt.Axis(format="d", tickMinStep=5))
    y = alt.Y("ValueCr:Q", title="Portfolio value (₹ Cr, inflation-adjusted)", scale=alt.Scale(zero=True),
              axis=alt.Axis(format=",.0f"))
    tip = [alt.Tooltip("Path:N"), alt.Tooltip("Age:Q", title="Age", format="d"),
           alt.Tooltip("Calendar:N", title="Financial year (April start)" if country == "India" else "Calendar year"),
           alt.Tooltip("ValueTxt:N", title="Portfolio value (PV)")]
    if country == "USA":
        tip.append(alt.Tooltip("ValueUSD:N", title=f"USD @ ₹{USD_INR}"))
    base = alt.Chart(df).encode(x=x, y=y, color=color, tooltip=tip)
    # The filled areas overlap, so they must not own the tooltip: hovering near one path would otherwise report the other.
    area = base.mark_area(opacity=0.15).encode(y=alt.Y("ValueCr:Q", stack=None), tooltip=alt.value(None))
    line = base.mark_line(strokeWidth=3)
    # Invisible, larger hover targets on every data point: each point reports its own path.
    hover = base.mark_point(size=450, filled=True, opacity=0)
    last = df.loc[df.groupby("Path")["Age"].idxmax()]
    # Suppress endpoint markers and value labels for paths that finish at zero.
    last = last.loc[last["ValueCr"] > 0]
    ends = alt.Chart(last).mark_point(size=130, filled=True, stroke="white", strokeWidth=1.5).encode(x=x, y=y, color=color, tooltip=tip)
    labels = alt.Chart(last).mark_text(align="left", dx=9, dy=-10, fontSize=12, fontWeight="bold").encode(
        x=x, y=y, text="EndLabel:N", color=alt.Color("EndColor:N", scale=None)
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#dc2626", strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q")
    one = alt.Chart(pd.DataFrame({"y": [float(corpus)]})).mark_rule(color="#94a3b8", strokeDash=[2, 3]).encode(y="y:Q")
    st.markdown("<div style='display:flex;gap:28px;flex-wrap:wrap;margin:.2rem 0 .4rem'>" + "".join(
        f"<span style='display:inline-flex;align-items:center;gap:8px'>"
        f"<span style='display:inline-block;width:30px;height:5px;border-radius:3px;background:{PALETTE[i]}'></span>"
        f"<span style='color:{PALETTE[i]};font-weight:500;font-size:.95rem'>{n}</span></span>"
        for i, n in enumerate(names)) + "</div>", unsafe_allow_html=True)
    # Extra right padding: Vega measures the end-label width slightly short, which clipped the last character(s).
    st.altair_chart((area + line + hover + ends + labels + zero + one).properties(
        height=340, width="container", padding={"left": 5, "top": 5, "right": 40, "bottom": 5}))



# ======================================================================
# MONTE CARLO: engine and page
# ======================================================================

# Colour thresholds for success probability (percent). Edit to taste.
GREEN_AT = 85.0
SUCCESS_ONLY_BELOW = 90.0   # show the survivors-only median only when success is below this (%)
AMBER_AT = 70.0

T_DF = 5  # degrees of freedom for fat-tailed equity returns

Y_CAP_CR = 2500.0  # upper limit of the trajectory chart's y-axis, in ₹ crore (future value)

START_AGE = 60      # age at the start of the first retirement year (change here if needed)
RANDOM_SEED = 42    # fixed seed: the same inputs always give the same results

# Guyton–Klinger guardrails are offered when the success probability is below this level (percent).
GK_TRIGGER_PCT = 75.0
# Original Guyton–Klinger: the capital-preservation (cut) rule is not applied in the final 15 years.
GK_NO_CUT_FINAL_YEARS = 15

NONE_LABEL = "None (standard Monte Carlo)"
GFC_LABEL = "2008–2012 (Global Financial Crisis & recovery)"
HISTORICAL_SEQUENCES = {GFC_LABEL: [-0.51, 0.78, 0.19, -0.24, 0.29]}


@dataclass(frozen=True)
class Inputs:
    corpus: float
    years: int
    start_age: int
    equity_pct: float
    rebalance_frequency: int
    equity_mean: float
    equity_vol: float
    fat_tails: bool
    debt_return: float
    inflation: float
    inflation_vol: float
    essential: float
    discretionary: float
    medical: float
    medical_premium: float
    safe_years: float
    bucket_refill: bool
    taper_age: Optional[int]
    taper_pct: float
    taper_essential: bool
    taper_discretionary: bool
    bad_year_cut_pct: float
    historical_initial_sequence: str
    stress_extra_inflation: float
    sims: int
    seed: int
    # Guyton–Klinger guardrails (all off by default; see run_simulation for the exact rules)
    gk_enabled: bool = False
    gk_upper: float = 0.20   # cut when withdrawal rate is this far ABOVE the starting rate
    gk_lower: float = 0.20   # raise when withdrawal rate is this far BELOW the starting rate
    gk_cut: float = 0.10     # size of each cut
    gk_raise: float = 0.10   # size of each raise
    gk_skip_inflation: bool = True
    gk_floor: float = 0.0    # hard floor on essential + discretionary spending, as a fraction of plan (real)


def validate_inputs(x: Inputs) -> None:
    finite_values = [
        x.corpus, x.equity_pct, x.equity_mean, x.equity_vol, x.debt_return,
        x.inflation, x.inflation_vol, x.essential, x.discretionary, x.medical,
        x.medical_premium, x.safe_years, x.taper_pct, x.bad_year_cut_pct,
        x.stress_extra_inflation,
    ]
    if not all(np.isfinite(v) for v in finite_values):
        raise ValueError("All numeric inputs must be finite.")
    if x.corpus < 0 or x.essential < 0 or x.discretionary < 0 or x.medical < 0:
        raise ValueError("Corpus and spending must be non-negative.")
    if not 0 <= x.equity_pct <= 1:
        raise ValueError("Equity allocation must be between 0 and 100%.")
    if x.rebalance_frequency not in (1, 3, 5):
        raise ValueError("Rebalancing frequency must be annual, every 3 years, or every 5 years.")
    if x.equity_vol < 0 or x.inflation_vol < 0:
        raise ValueError("Volatility cannot be negative.")
    if x.inflation <= -1 or x.debt_return <= -1 or x.medical_premium <= -1:
        raise ValueError("Inflation, debt return and medical premium must be greater than -100%.")
    if not 0 <= x.taper_pct < 1:
        raise ValueError("Taper must be between 0% and less than 100%.")
    if not 0 <= x.bad_year_cut_pct < 1:
        raise ValueError("Discretionary cut must be between 0% and less than 100%.")
    if x.years < 1 or x.start_age < 0:
        raise ValueError("Retirement years must be positive and age non-negative.")
    if x.safe_years < 0:
        raise ValueError("Safe bucket years cannot be negative.")
    if x.sims < 1:
        raise ValueError("Simulation count must be at least 1.")
    if not all(np.isfinite(v) for v in (x.gk_upper, x.gk_lower, x.gk_cut, x.gk_raise)):
        raise ValueError("Guardrail settings must be finite.")
    if x.gk_upper < 0 or x.gk_lower < 0 or x.gk_lower >= 1:
        raise ValueError("Guardrail bands must be non-negative (lower band below 100%).")
    if not 0 <= x.gk_cut < 1 or x.gk_raise < 0:
        raise ValueError("Guardrail cut must be 0% to under 100%, and the raise cannot be negative.")
    if not np.isfinite(x.gk_floor) or not 0 <= x.gk_floor <= 1:
        raise ValueError("Spending floor must be between 0% and 100% of plan.")


def spending_for_year(
    x: Inputs, age: int, core_factor: np.ndarray, medical_factor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs are first-retirement-year amounts; factors are path-specific price levels.
    core_factor scales essential and discretionary spending (it equals the general price level
    unless the guardrail inflation-skip rule has frozen it); medical_factor scales medical."""
    ess = x.essential * core_factor
    disc = x.discretionary * core_factor
    med = x.medical * medical_factor

    if x.taper_age is not None and age >= x.taper_age:
        taper_years = age - x.taper_age + 1
        taper_factor = (1 - x.taper_pct) ** taper_years
        if x.taper_essential:
            ess = ess * taper_factor
        if x.taper_discretionary:
            disc = disc * taper_factor
    return ess, disc, med


def run_simulation(x: Inputs) -> dict:
    """
    Vectorised Monte Carlo engine.
    Order within a year: returns -> (periodic) rebalance -> withdrawal -> (refill) -> inflation draw.
    If a withdrawal cannot be fully funded, that path fails permanently.
    Ending at exactly zero after the final fully funded withdrawal is success.

    Guardrails (Guyton–Klinger, when x.gk_enabled), applied to essential + discretionary spending only
    (medical is never cut). Withdrawal rate = this year's planned spending / portfolio value at the start of
    the year (safe bucket included). "Initial rate" = first-year spending / starting corpus.
      * Capital preservation: rate > initial x (1 + upper band) -> cut spending by gk_cut. Not applied in the
        final GK_NO_CUT_FINAL_YEARS years of the horizon.
      * Prosperity: rate < initial x (1 - lower band) -> raise spending by gk_raise.
      * Inflation rule (optional): after a year with a negative portfolio return, skip the inflation increase
        when holding spending flat would still leave the rate above the initial rate; the missed increase is
        never made up.
      * Floor (optional): essential + discretionary spending is never allowed below gk_floor x plan (today's ₹).
    With guardrails on, the separate "discretionary cut in negative-return years" is not applied.
    """
    validate_inputs(x)
    n, years = x.sims, x.years
    rng = np.random.default_rng(x.seed)

    first_year_spend = x.essential + x.discretionary + x.medical
    safe0 = min(x.corpus, x.safe_years * first_year_spend)
    invested0 = x.corpus - safe0

    safe = np.full(n, safe0, dtype=float)
    equity = np.full(n, invested0 * x.equity_pct, dtype=float)
    debt = np.full(n, invested0 * (1.0 - x.equity_pct), dtype=float)

    success = np.ones(n, dtype=bool)
    failure_year = np.full(n, -1, dtype=int)  # 0-based; -1 = successful
    unfunded_first = np.zeros(n, dtype=float)
    inflation_factor = np.ones(n, dtype=float)
    medical_factor = np.ones(n, dtype=float)
    core_factor = np.ones(n, dtype=float)   # price level for essential + discretionary spending
    gk_mult = np.ones(n, dtype=float)       # cumulative guardrail cuts/raises
    cut_count = np.zeros(n, dtype=int)
    raise_count = np.zeros(n, dtype=int)
    spend_ratio = np.full((years, n), np.nan) if x.gk_enabled else None  # core spending vs plan (real)
    e0, d0, m0 = spending_for_year(x, x.start_age, np.ones(1), np.ones(1))
    iwr = float((e0 + d0 + m0)[0] / x.corpus) if x.corpus > 0 else 0.0  # initial withdrawal rate

    hist = HISTORICAL_SEQUENCES.get(x.historical_initial_sequence)
    hist_len = len(hist) if hist is not None else 0
    t_scale = math.sqrt(T_DF / (T_DF - 2))

    # Total balance in today's rupees (deflated by the path's own price level).
    real_balances = np.zeros((years + 1, n), dtype=float)
    real_balances[0] = safe + equity + debt
    nominal_balances = np.zeros((years + 1, n), dtype=float)
    nominal_balances[0] = safe + equity + debt

    for yr in range(years):
        age = x.start_age + yr
        active = success.copy()
        start_total = safe + equity + debt  # portfolio at the start of the year, before returns

        # ---- equity return draw ----
        if yr < hist_len:
            # Historical stress window: same observed return on every path; random draws resume after.
            eq_r = np.full(n, hist[yr], dtype=float)
        else:
            if x.fat_tails:
                z = rng.standard_t(T_DF, n) / t_scale
            else:
                z = rng.standard_normal(n)
            eq_r = x.equity_mean + x.equity_vol * z
        eq_r = np.maximum(eq_r, -1.0)

        # ---- apply returns ----
        eq_pre, debt_pre = equity.copy(), debt.copy()
        equity[active] *= 1.0 + eq_r[active]
        debt[active] *= 1.0 + x.debt_return
        safe[active] *= 1.0 + x.debt_return

        inv_pre = eq_pre + debt_pre
        portfolio_r = np.zeros(n, dtype=float)
        pos = inv_pre > 0
        portfolio_r[pos] = (eq_pre[pos] * eq_r[pos] + debt_pre[pos] * x.debt_return) / inv_pre[pos]

        # ---- periodic rebalance of invested assets (after returns, before withdrawal) ----
        if (yr + 1) % x.rebalance_frequency == 0:
            inv = equity + debt
            equity[active] = inv[active] * x.equity_pct
            debt[active] = inv[active] * (1.0 - x.equity_pct)

        # ---- scheduled spending ----
        ess, disc, med = spending_for_year(x, age, core_factor, medical_factor)
        down = (portfolio_r < 0.0) & active
        if x.gk_enabled:
            start_wr = np.divide(
                (ess + disc) * gk_mult + med, start_total,
                out=np.full(n, np.inf), where=start_total > 0,
            )
            can_cut = (years - yr) > GK_NO_CUT_FINAL_YEARS
            hit_up = active & can_cut & (start_wr > iwr * (1.0 + x.gk_upper))
            hit_dn = active & ~hit_up & (start_wr < iwr * (1.0 - x.gk_lower))
            m_prev = gk_mult
            m_new = np.where(hit_up, m_prev * (1.0 - x.gk_cut), m_prev)
            m_new = np.where(hit_dn, m_new * (1.0 + x.gk_raise), m_new)
            if x.gk_floor > 0:
                # Hard floor: essential + discretionary spending never falls below gk_floor x plan, in
                # today's rupees. This also stops skipped inflation increases from pushing it below.
                m_new = np.maximum(m_new, x.gk_floor * inflation_factor / core_factor)
            cut_count += hit_up & (m_new < m_prev - 1e-12)      # only cuts that actually lowered spending
            raise_count += hit_dn & (m_new > m_prev + 1e-12)
            gk_mult = m_new
            ess = ess * gk_mult
            disc = disc * gk_mult
            spend_ratio[yr] = np.where(active, gk_mult * core_factor / inflation_factor, np.nan)
        else:
            disc = np.where(down, disc * (1.0 - x.bad_year_cut_pct), disc)
        scheduled = ess + disc + med

        # ---- withdrawal ----
        # Refill mode: bucket first in down years, invested first otherwise (bucket as backstop).
        # Original mode: bucket first every year.
        inv = equity + debt
        safe_first = down if x.bucket_refill else np.ones(n, dtype=bool)
        d_safe1 = np.where(safe_first, np.minimum(safe, scheduled), 0.0)
        d_inv1 = np.where(safe_first, 0.0, np.minimum(inv, scheduled))
        rem = scheduled - d_safe1 - d_inv1
        d_inv2 = np.minimum(inv - d_inv1, rem)
        rem = rem - d_inv2
        d_safe2 = np.minimum(safe - d_safe1, rem)
        rem = rem - d_safe2
        shortfall = np.maximum(rem, 0.0)
        safe_draw = d_safe1 + d_safe2
        inv_draw = d_inv1 + d_inv2

        eq_share = np.divide(equity, inv, out=np.zeros(n), where=inv > 0)
        eq_draw = inv_draw * eq_share
        safe[active] -= safe_draw[active]
        equity[active] -= eq_draw[active]
        debt[active] -= (inv_draw - eq_draw)[active]

        failed_now = active & (shortfall > 1e-4)
        success[failed_now] = False
        failure_year[failed_now] = yr
        unfunded_first[failed_now] = shortfall[failed_now]

        # ---- bucket refill after non-down years ----
        if x.bucket_refill and x.safe_years > 0:
            refill_ok = active & ~failed_now & ~down
            inv = equity + debt
            need = np.maximum(x.safe_years * scheduled - safe, 0.0)
            move = np.where(refill_ok, np.minimum(need, inv), 0.0)
            share = np.divide(equity, inv, out=np.zeros(n), where=inv > 0)
            equity -= move * share
            debt -= move * (1.0 - share)
            safe += move

        safe = np.maximum(safe, 0.0)
        equity = np.maximum(equity, 0.0)
        debt = np.maximum(debt, 0.0)
        total = safe + equity + debt

        # ---- inflation realised during this year; sets next year's price levels ----
        stress_extra = x.stress_extra_inflation if yr < hist_len else 0.0
        infl = np.maximum(rng.normal(x.inflation + stress_extra, x.inflation_vol, n), -0.99)
        growth = 1.0 + infl
        inflation_factor *= growth
        medical_factor *= growth * (1.0 + x.medical_premium)
        if x.gk_enabled and x.gk_skip_inflation:
            # Inflation rule: after a down year, hold spending flat if that still leaves the withdrawal
            # rate above the initial rate. The skipped increase is not made up later.
            wr_flat = np.divide(scheduled, total, out=np.full(n, np.inf), where=total > 0)
            skip = down & (wr_flat > iwr)
            core_factor *= np.where(skip, 1.0, growth)
        else:
            core_factor *= growth

        real_balances[yr + 1] = total / inflation_factor
        nominal_balances[yr + 1] = total

    return {
        "real_balances": real_balances,
        "nominal_balances": nominal_balances,
        "success": success,
        "failure_year": failure_year,
        "unfunded_first": unfunded_first,
        "success_rate": float(success.mean()),
        "terminal_real": real_balances[-1],
        "initial_safe": safe0,
        "initial_invested": invested0,
        "first_year_spend": first_year_spend,
        "spend_ratio": spend_ratio,
        "cut_count": cut_count,
        "raise_count": raise_count,
        "initial_withdrawal_rate": iwr,
    }


def funded_through_age(result: dict, start_age: int, age: int) -> float:
    fy = result["failure_year"]
    failed_by = (fy >= 0) & (start_age + fy <= age)
    return float(1.0 - failed_by.mean())


def legacy_amount(real_balances: np.ndarray, idx: int, confidence_pct: float) -> float:
    """Amount (today's rupees) that at least `confidence_pct`% of simulated paths leave behind at row idx.
    Uses the (100 - confidence) percentile with 'lower' interpolation, so the figure is an observed outcome
    and is not rounded up."""
    return float(np.percentile(real_balances[idx], 100.0 - confidence_pct, method="lower"))


def fmt_money(v: float) -> str:
    if abs(v) >= 1e7:
        return f"₹{v/1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:,.2f} L"
    return f"₹{v:,.0f}"


def status_class(pct: float) -> str:
    if pct < AMBER_AT:
        return "success-red"
    if pct < GREEN_AT:
        return "success-yellow"
    return "success-green"


def status_icon(pct: float) -> str:
    """Traffic-light marker for tables, using the same thresholds as status_class."""
    if pct != pct:  # NaN
        return "—"
    if pct < AMBER_AT:
        return "🔴"
    if pct < GREEN_AT:
        return "🟡"
    return "🟢"


def kpi_card(label: str, value_html: str, value_class: str, last: bool = False) -> str:
    last_class = " last" if last else ""
    return (
        f'<div class="retirement-kpi{last_class}">'
        f'<div class="retirement-kpi-label">{label}</div>'
        f'<div class="retirement-kpi-value {value_class}">{value_html}</div>'
        f'</div>'
    )


@st.cache_data(show_spinner="Running simulation…")
def cached_run(params: dict) -> dict:
    return run_simulation(Inputs(**params))


@st.cache_data(show_spinner=False)
def cached_success(params: dict) -> float:
    return run_simulation(Inputs(**params))["success_rate"]


def sensitivity_table(x: Inputs) -> pd.DataFrame:
    """One-at-a-time shocks, same seed (common random numbers), capped at 10,000 sims per row."""
    base = replace(x, sims=min(x.sims, 10000))

    def clamp(v: float) -> float:
        return min(1.0, max(0.0, v))

    scenarios = [
        ("Base case", {}),
        ("Equity return −2 pp", {"equity_mean": x.equity_mean - 0.02}),
        ("Equity return +2 pp", {"equity_mean": x.equity_mean + 0.02}),
        ("Inflation +1 pp", {"inflation": x.inflation + 0.01}),
        ("Inflation −1 pp", {"inflation": x.inflation - 0.01}),
        ("Equity volatility +5 pp", {"equity_vol": x.equity_vol + 0.05}),
        ("Debt return −1 pp", {"debt_return": x.debt_return - 0.01}),
        ("Spending +10%", {
            "essential": x.essential * 1.1,
            "discretionary": x.discretionary * 1.1,
            "medical": x.medical * 1.1,
        }),
        ("Equity allocation −10 pp", {"equity_pct": clamp(x.equity_pct - 0.10)}),
        ("Equity allocation +10 pp", {"equity_pct": clamp(x.equity_pct + 0.10)}),
    ]
    rows = []
    for label, change in scenarios:
        try:
            rate = cached_success(asdict(replace(base, **change))) * 100.0
        except ValueError:
            rate = float("nan")
        rows.append({"Scenario": label, "Success (%)": rate})
    df = pd.DataFrame(rows)
    df["Change vs base (pp)"] = df["Success (%)"] - df.loc[0, "Success (%)"]
    df["Success (%)"] = df["Success (%)"].round(1)
    df["Change vs base (pp)"] = df["Change vs base (pp)"].round(1)
    df.insert(1, "Status", [status_icon(v) for v in df["Success (%)"]])
    return df


def floor_scan(gk_inputs: Inputs) -> list:
    """Success (%) at each spending floor from 30% to 100% of plan, other guardrail settings unchanged.
    Uses up to 10,000 simulations per floor and the same random numbers for every floor."""
    base = replace(gk_inputs, sims=min(gk_inputs.sims, 10000))
    return [
        (f, cached_success(asdict(replace(base, gk_floor=f / 100.0))) * 100.0)
        for f in range(30, 101, 5)
    ]


def alert(level: str, text: str) -> None:
    """Rounded, pale-background alert box. level: error, warning, caution, success or info; error and warning are red, caution is amber, success is green, info (neutral figures) is blue-grey.
    **bold** in the text is shown bold."""
    safe = html.escape(str(text), quote=False)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    st.markdown(f'<div class="rm-alert rm-alert-{level}">{safe}</div>', unsafe_allow_html=True)


KPI_CSS = """
<style>
.retirement-kpi {
    min-height: 150px;
    padding: 10px 14px 12px 10px;
    border-right: 2px solid #D6E0EC;
    box-sizing: border-box;
}
.retirement-kpi.last { border-right: none; }
.retirement-kpi-label {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.25;
    color: #185AAB;
    min-height: 56px;
    margin-bottom: 8px;
    overflow-wrap: break-word;
}
.retirement-kpi-value {
    font-size: 24px;
    font-weight: 500;
    line-height: 32px;
    color: #252936;
    white-space: normal;
    overflow-wrap: anywhere;
    font-variant-numeric: tabular-nums;
}
.retirement-kpi-value.success-value { font-weight: 800; }
.retirement-kpi-value.success-red { color: #C62828; }
.retirement-kpi-value.success-yellow { color: #B77900; }
.retirement-kpi-value.success-green { color: #16803C; }
.retirement-kpi-subvalue {
    display: block;
    font-size: 20px;
    line-height: 32px;
    font-weight: 400;
    white-space: nowrap;
}
.retirement-kpi-subvalue.real-value { margin-top: 4px; }
.retirement-kpi-subvalue.corpus-positive, .retirement-kpi-value.safe-positive {
    color: #16803C;
    font-weight: 800;
}
.retirement-kpi-subvalue.corpus-zero, .retirement-kpi-value.safe-zero {
    color: #C62828;
    font-weight: 800;
}
/* Guardrails section: every percentage shares one size */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-size: 24px !important; line-height: 32px !important; font-weight: 800 !important; color: #252936; font-variant-numeric: tabular-nums; }
.retirement-kpi-subvalue.gk-pp { font-size: 24px; font-weight: 800; }
@media (max-width: 1400px) {
    .retirement-kpi-label { min-height: 84px; }
}
@media (max-width: 1200px) {
    .retirement-kpi-subvalue { white-space: normal; }
}
@media (max-width: 1100px) {
    .retirement-kpi { padding: 10px 9px 12px 8px; }
    .retirement-kpi-label { font-size: 19px; min-height: 72px; }
    .retirement-kpi-value { font-size: 21px; }
    .retirement-kpi-subvalue { font-size: 18px; }
}
@media (max-width: 900px) {
    .retirement-kpi { border-right: none; border-bottom: 1px solid #D6E0EC; padding: 12px 8px; }
    .retirement-kpi-label { min-height: 0; }
}
.rm-alert {
    border-radius: 12px;
    padding: 18px 32px;
    margin: 0 0 1rem 0;
    font-size: 1.05rem;
    line-height: 1.5;
}
.rm-alert strong { font-weight: 700; }
.rm-alert-error, .rm-alert-warning { background: #FDE2E4; color: #B3261E; }
.rm-alert-success { background: #DDF3E4; color: #1B6B3A; }
.rm-alert-caution { background: #FFF4DB; color: #6B4E0A; }
.rm-alert-info { background: #E8EEF7; color: #1F3A5F; }
.rm-failed-row { display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 1.1rem; row-gap: .3rem; margin: .2rem 0 1.2rem 0; color: #475569; font-size: .98rem; }
.rm-failed-row strong { color: #1E293B; font-weight: 700; }
.rm-failed-row strong.rm-failed-count { color: #B5483F; }
.rm-failed-sep { color: #94A3B8; }
</style>
"""


def explanation_topics(inputs: Inputs, result: dict, ss) -> dict:
    """Plain-language explanations shown in the 'How to read this page' dropdown.
    The guardrail examples use your own corpus, spending and slider settings."""
    corpus = inputs.corpus
    iwr = float(result["initial_withdrawal_rate"])
    total0 = iwr * corpus                      # first-year spending (includes medical if it is in the plan)
    core0 = inputs.essential + inputs.discretionary
    up, lo = ss["gk_upper_pct"] / 100.0, ss["gk_lower_pct"] / 100.0
    cut, rais = ss["gk_cut_pct"] / 100.0, ss["gk_raise_pct"] / 100.0
    floor = int(ss["gk_floor_pct"])
    up_rate, lo_rate = iwr * (1 + up) * 100.0, iwr * (1 - lo) * 100.0
    p_up, p_lo = corpus / (1 + up), corpus / (1 - lo)          # portfolio that trips each guardrail
    fall_pct, grow_pct = up / (1 + up) * 100.0, lo / (1 - lo) * 100.0
    medical_note = (
        "\n\nYour medical spending is part of the rate, but cuts, raises and the floor apply to essential "
        "and discretionary spending only." if inputs.medical > 0 else ""
    )
    return {
        "Probability of Success": f"""
**What it is:** out of all the simulated futures, the share in which *every* year's spending was paid in full until the end of the horizon.

**How to read it:** it is not a forecast. It tells you how often *this plan, under these assumptions,* works.

**Colours:** 🟢 green = {GREEN_AT:.0f}% or more · 🟡 amber = {AMBER_AT:.0f}% to {GREEN_AT:.0f}% · 🔴 red = below {AMBER_AT:.0f}%.

**Example:** 80% means that in 8 of 10 simulated futures the money lasted, and in 2 of 10 it ran out before the end.
""",
        "Funded Through Age (e.g. 90)": """
The same idea as Probability of Success, but only up to that age.

A plan can be very safe to age 90 and still fail in the 100s. If this figure is much higher than the full-horizon success, the risk sits in the very late years.
""",
        "Median Corpus at End": """
**Median** means the middle outcome: line up all the simulated futures from poorest to richest, and the median is the one in the middle. Half end with more, half with less. It is shown in **today's rupees**.

- **All paths:** the middle outcome of a randomly drawn future. Futures where the money ran out count as ₹0. If success is below 50%, this is ₹0, because more than half the futures ended with nothing. This is the honest headline figure.

Use the **Legacy** section for the amount you can count on.
""",
        "Initial Safe Bucket": """
Money set aside at the start, in low-risk debt, equal to the number of years of spending you choose (default 5). It earns the debt return.

- By default it is **spent first** each year and is **not refilled**, so it is usually gone after about five years. It protects the early years from forced selling during a crash, but not later years.
- A refill option is available in the sidebar (Risk protection section). In my testing it lowered success, because it keeps years of spending in a low-return asset.
- It reduces the money invested in equities, so equity as a share of the **whole** corpus is lower than your equity slider.
""",
        "Summary bar (corpus, bucket, withdrawal rate)": """
- **Initial corpus:** your starting money.
- **Safe bucket / Invested:** how that money is split at the start.
- **Equity as share of whole corpus:** your equity % applies to the invested part only, so the overall share is lower.
- **First-year spending and % of corpus:** this percentage is your **starting withdrawal rate**. For a long horizon, a high rate (for example 5% or more) is a warning sign.
- **Expected invested-portfolio return:** the weighted average of your equity and debt return assumptions, before inflation and before withdrawals.
""",
        "Failed paths": """
This line appears under the summary bar when some simulated futures run out of money.

- **X of Y paths failed (Z%):** X is how many of the Y simulated futures (paths) ran out of money, meaning some year's spending could not be paid in full. X ÷ Y is the failure rate Z, and 100% minus that is the Probability of Success.
""",
        "Portfolio Trajectories chart": """
- **x-axis:** your age. **y-axis:** portfolio value in **future (nominal) rupees**, meaning the rupees of that future year.
- **Title:** the median ending corpus at the final age, shown as **present value** in today's rupees, meaning what it is worth in today's purchasing power. The chart itself plots future rupees, which look large because of inflation.
- **Dark blue line:** the median. **Darker band:** the middle 50% of outcomes. **Lighter band:** the middle 80%.
- **Thin light-blue lines:** up to 100 individual simulated futures. Paths that go above the y-axis limit are left out of this sample.
- **Y-axis limit:** the chart stops at ₹2,500 Cr (or lower when the data is smaller). The statistics use every simulation.
- Lines dropping to zero are futures where the money ran out.
""",
        "Ruin probability and 'fully funded through age'": f"""
**Ruin probability by age:** the running total of futures that have already run out of money by that age. It only goes up. A steep part of the line marks the risky years.

**Fully funded through age table:** the share of futures still fully funded at each age. 🟢🟡🔴 uses the same thresholds as Probability of Success ({GREEN_AT:.0f}% and {AMBER_AT:.0f}%).

A failure means that in some year the full spending could not be paid.
""",
        "Legacy (what you leave behind)": """
**"With 90% confidence you leave at least ₹X"** means: in 90% of the simulated futures, the amount left at the chosen age is ₹X or more. In the other 10% it is less, including ₹0.

- Higher confidence gives a **smaller** amount. That is the trade-off.
- Amounts are in today's rupees.
- If the plan often fails, the amount at high confidence is ₹0.
""",
        "Guardrails (Guyton–Klinger): the idea": f"""
Shown when success is below {GK_TRIGGER_PCT:.0f}%. Instead of spending your plan blindly, you adjust spending by simple rules. The figures below use **your** corpus, spending and current guardrail settings.

**Withdrawal rate = planned spending ÷ portfolio at the start of the year.** It is not a return.

**Your plan:** {fmt_money(total0)} a year on a {fmt_money(corpus)} portfolio is a **starting rate of {iwr * 100:.2f}%**. Every rule compares the current rate with this starting rate.

- **Cut:** the upper guardrail is {up * 100:.0f}% above the start, so a rate above **{up_rate:.2f}%**. With spending at plan, that happens if your portfolio (in today's rupees) falls to about **{fmt_money(p_up)}**, which is {fall_pct:.0f}% lower. Spending is then cut by {cut * 100:.0f}%: {fmt_money(core0)} becomes {fmt_money(core0 * (1 - cut))}.
- **Raise:** the lower guardrail is {lo * 100:.0f}% below the start, so a rate below **{lo_rate:.2f}%**. That happens if your portfolio grows to about **{fmt_money(p_lo)}**, which is {grow_pct:.0f}% higher. Spending is then raised by {rais * 100:.0f}%: {fmt_money(core0)} becomes {fmt_money(core0 * (1 + rais))}.
- **Floor:** spending never goes below {floor}% of plan, which is {fmt_money(core0 * floor / 100.0)} a year in today's rupees.{medical_note}

The rules are checked **once at the start of each year**, so spending changes at most once a year (one cut or one raise). Guardrails replace the "discretionary cut in negative years" setting while they are on.
""",
        "Guardrail sliders": f"""
- **Apply guardrails:** switch on or off to compare.
- **Upper guardrail (now {up * 100:.0f}%):** how far the withdrawal rate can rise above your starting rate ({iwr * 100:.2f}%) before you cut. For your plan, that is a rate above {up_rate:.2f}%.
- **Cut spending by (now {cut * 100:.0f}%):** the size of each cut. {fmt_money(core0)} becomes {fmt_money(core0 * (1 - cut))}.
- **Lower guardrail (now {lo * 100:.0f}%):** how far the rate can fall below your starting rate before you raise spending. For your plan, that is a rate below {lo_rate:.2f}%.
- **Raise spending by (now {rais * 100:.0f}%):** the size of each raise. {fmt_money(core0)} becomes {fmt_money(core0 * (1 + rais))}.
- **Skip inflation after a negative-return year:** after a year when the portfolio fell, spending is not increased for inflation (if the rate is still above the starting rate). The skipped increase is never made up.
- **Spending floor (now {floor}%):** a hard limit. Spending in today's rupees never goes below {fmt_money(core0 * floor / 100.0)} a year (it grows with inflation in future rupees). If cutting to the floor is not enough, the money runs out.
- **Cuts stop in the final 15 years** of the horizon, as in the original rules. Raises still apply.
- **How often:** all rules are tested once a year, so at most one cut or one raise happens in any single year.

**The trade-off:** a tighter upper guardrail, a bigger cut or a higher floor changes how much lifestyle you give up versus how safe the plan becomes.
""",
        "Guardrail results": f"""
Every figure here is about **paths**. A path is one simulated future: one complete run of your retirement from the first year to the end of the horizon ({inputs.years} years).

- **Success without / with guardrails:** the share of paths where every year's spending was paid in full, without and with the rules. Coloured with the same thresholds ({GREEN_AT:.0f}% and {AMBER_AT:.0f}%). The "pp vs without" line is the gain in percentage points.
- **Paths with at least one spending cut:** out of all simulated paths, the share in which spending was cut in at least one year. "At least one" means one or more cut years, whether 1 or 10. Example: 72% means 72 of every 100 simulated futures needed one or more cuts, and the other 28 never did. It does not say how many cuts or how deep; the median and bad-case spending figures below show that.
- **Paths that reach your spending floor:** the share of paths where cuts pushed spending all the way down to the floor in at least one year. A high figure means the plan is tight.
- **Median spending vs plan:** for each path, its average spending as a percentage of your plan (in today's rupees) over the years it was funded. The median is the middle path, so it is the typical outcome and is not pulled up by the few lucky paths that spend far above plan. 85% means the typical path averaged 85% of the planned lifestyle.
- **Bad-case spending vs plan (10th percentile):** 1 in 10 paths averaged less than this. It shows how low a bad but plausible future goes.

If success with guardrails stays low even with your floor in place, the fix is a bigger corpus, lower planned spending or a shorter horizon.
""",
        "Floor check (recommended floor)": f"""
The app tries every spending floor from 30% to 100% of plan, in steps of 5, keeping your other guardrail settings, and finds the **highest floor that still reaches {GREEN_AT:.0f}% success** (the green level).

- **Your floor is lower than that:** you can afford to protect more spending. It suggests raising your floor to the highest one that still reaches the target.
- **Your floor is too high:** success falls below {GREEN_AT:.0f}%, so the floor is too optimistic. It suggests lowering it to the highest floor that works.
- **Within 5 points:** your floor is about right.
- **No floor reaches the target:** guardrails alone cannot fix the plan. Change the corpus, spending or horizon.

Each floor uses up to 10,000 simulations, so figures can differ a little from the headline. The "Success at each spending floor" table shows every floor.
""",
        "Spending under guardrails chart": """
- **Grey dashed line (100%):** your plan. **Red line:** your spending floor.
- **Blue line:** typical (median) spending as a share of your plan, in today's rupees. **Band:** the middle 80% of paths.
- **Below 100%:** typical futures involve cuts. **Above 100%:** raises happened.
- Only paths that are still funded are included, so the later ages describe the survivors.
""",
        "Sensitivity table": f"""
Changes **one assumption at a time** (equity return, inflation, volatility, debt return, spending, equity share) with the same random numbers, and shows the new success rate. 🟢🟡🔴 uses the same thresholds ({GREEN_AT:.0f}% and {AMBER_AT:.0f}%).

Large swings mean the headline number depends heavily on that assumption, so treat it with caution. Small swings mean the result is more robust.
""",
        "Inputs: returns and inflation": f"""
- **Expected equity return:** the average yearly return. **Volatility:** how widely yearly returns swing around it.
- **Fat-tailed returns (on by default):** extreme years (big crashes and spikes) occur more often than a normal bell curve allows, using a Student-t distribution with {T_DF} degrees of freedom and the same average and volatility. It gives lower, more cautious success rates.
- **Debt return:** fixed every year. **Inflation:** average and volatility. If debt return is below inflation, debt loses purchasing power by construction, and the app warns you.
- **Rebalancing:** how often equity and debt are restored to your target mix.
""",
        "Inputs: spending, medical and safe bucket": """
- **Essential / discretionary spend:** first-year amounts, then raised with inflation.
- **Separate medical corpus planned?** Yes: medical costs are left out entirely, so enter only your non-medical corpus. No: a medical spending line is added that grows faster than general inflation (inflation plus a premium) and is never tapered or cut.
- **Taper:** from the chosen age, real spending falls by the chosen percentage each year.
- **Discretionary cut in negative years:** discretionary spending is reduced after a year when the invested portfolio fell (replaced by guardrails when those are on).
- **Safe bucket:** see "Initial Safe Bucket".
""",
        "Inputs: stress test and simulation": """
- **Historical initial-market stress:** the first five retirement years use the 2008–2012 sequence set in the app instead of random returns (about −51%, +78%, +19%, −24%, +29%), then random returns resume. It tests a bad start, which is the most damaging time for a crash. You can add extra inflation during those years.
- **Number of simulations:** fixed at 100,000 paths for a steady result.
- **Repeatable results:** the simulation uses a fixed random seed, so the same inputs always give the same results. Differences of less than about 0.4 percentage points between two settings can be sampling noise.
""",
        "Limits to keep in mind": """
- **Not modelled:** taxes, fees, other income (rent, pension, annuity), one-off shocks (for example a large medical event or property purchase) and any legacy target.
- **Each year is independent:** returns do not cluster into good and bad runs, and inflation is unrelated to returns. Real markets are messier.
- **Debt return is fixed.** There is no interest-rate or credit risk.
- **Ages assume retirement starts at 60.** Ages on the charts, the taper age and the legacy age are measured from that start.
- **Results are probabilities from your assumptions, not predictions.** Change the assumptions and the answer changes. Use the sensitivity table to see by how much.
- **Success means every year's spending was paid in full** (with guardrails on, the adjusted spending, never below your floor).
""",
    }


REBALANCE_OPTIONS = {"Annually": 1, "Every 3 years": 3, "Every 5 years": 5}
SIM_OPTIONS = [1000, 5000, 10000, 25000, 50000, 100000]

# Default value of every input. Inputs live in st.session_state, so a value is kept even while its
# section is not on screen (only one sidebar section is shown at a time).
DEFAULTS = {
    "corpus_cr": 5.0, "years": 40,
    "essential_l": 12.0, "discretionary_l": 6.0, "medical_separate": "Yes",
    "medical_l": 3.0, "medical_premium_pct": 3.0,
    "taper_age": 80, "taper_pct": 2.0, "taper_essential": True, "taper_discretionary": True,
    "equity_mean_pct": 10.0, "equity_vol_pct": 15.0, "fat_tails": True,
    "debt_return_pct": 5.0, "inflation_pct": 6.0, "inflation_vol_pct": 2.0,
    "equity_pct": 60, "rebalance_label": "Annually",
    "safe_years": 5.0, "bucket_refill": False, "cut_pct": 30.0,
    "historical_sequence": NONE_LABEL, "stress_extra_pct": 0.0,
    "gk_on": True, "gk_upper_pct": 20, "gk_cut_pct": 10, "gk_lower_pct": 20, "gk_raise_pct": 10,
    "gk_skip": True, "gk_floor_pct": 75,
    "legacy_conf": 90, "legacy_age_in": 0,
    "sims": 100000,
}


def init_state() -> None:
    """Set defaults once, and re-save every value each run so it survives while its widget is hidden."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
        else:
            st.session_state[key] = st.session_state[key]


def build_inputs() -> Inputs:
    ss = st.session_state
    medical_on = ss["medical_separate"] == "No"
    stress = str(ss["historical_sequence"])
    return Inputs(
        corpus=float(ss["corpus_cr"]) * 1e7, years=int(ss["years"]), start_age=START_AGE,
        equity_pct=float(ss["equity_pct"]) / 100, rebalance_frequency=REBALANCE_OPTIONS[ss["rebalance_label"]],
        equity_mean=float(ss["equity_mean_pct"]) / 100, equity_vol=float(ss["equity_vol_pct"]) / 100,
        fat_tails=bool(ss["fat_tails"]), debt_return=float(ss["debt_return_pct"]) / 100,
        inflation=float(ss["inflation_pct"]) / 100, inflation_vol=float(ss["inflation_vol_pct"]) / 100,
        essential=float(ss["essential_l"]) * 1e5, discretionary=float(ss["discretionary_l"]) * 1e5,
        medical=float(ss["medical_l"]) * 1e5 if medical_on else 0.0,
        medical_premium=float(ss["medical_premium_pct"]) / 100 if medical_on else 0.0,
        safe_years=float(ss["safe_years"]), bucket_refill=bool(ss["bucket_refill"]),
        taper_age=None if int(ss["taper_age"]) == 0 else int(ss["taper_age"]),
        taper_pct=float(ss["taper_pct"]) / 100, taper_essential=bool(ss["taper_essential"]),
        taper_discretionary=bool(ss["taper_discretionary"]), bad_year_cut_pct=float(ss["cut_pct"]) / 100,
        historical_initial_sequence=stress,
        stress_extra_inflation=float(ss["stress_extra_pct"]) / 100 if stress == GFC_LABEL else 0.0,
        sims=int(ss["sims"]), seed=RANDOM_SEED,
    )


CORPUS_OPTS_MC = ([round(0.05 * k, 2) for k in range(1, 20)] + [round(1.0 + 0.1 * k, 1) for k in range(0, 90)]
                  + [round(10 + 0.25 * k, 2) for k in range(0, 361)])        # Cr: 5 L steps, then 0.1 Cr, then 0.25 Cr
SPEND_OPTS_MC = [round(0.5 * k, 1) for k in range(0, 301)]                    # lakh: 0 to 150 in 0.5 L steps


def _trim(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _fmt_cr_l(cr: float) -> str:
    """Below 1 Cr show lakh, from 1 Cr show crore."""
    return f"₹{_trim(cr * 100, 0)} L" if cr < 1 else f"₹{_trim(cr, 2)} Cr"


def _fmt_l_cr(lakh: float) -> str:
    return f"₹{_trim(lakh, 1)} L" if lakh < 100 else f"₹{_trim(lakh / 100, 2)} Cr"


def _exact_slider(label: str, key: str, options: list, fmt) -> None:
    """Slider that moves on a tidy grid but never alters the exact stored value unless you move it."""
    ss = st.session_state
    wkey = key + "_w"
    snap = min(options, key=lambda o: abs(o - float(ss[key])))
    if wkey not in ss or abs(float(ss[wkey]) - snap) > 1e-9:
        ss[wkey] = snap

    def _moved() -> None:
        ss[key] = float(ss[wkey])

    st.select_slider(label, options=options, key=wkey, format_func=fmt, on_change=_moved)


def _mc_slider_css() -> None:
    """Corpus and spend sliders: same look as the Historical Stress Test sliders (red thumb and fill, light-grey track)."""
    st.markdown("""<style>
.st-key-mc_cs [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] { background-color: #E53935 !important; }
.st-key-mc_cs [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background-color: #E53935 !important; border-color: #E53935 !important; }
.st-key-mc_cs [data-testid="stSlider"] [data-baseweb="slider"] > div > div { background-color: #DCE2EC; }
.st-key-mc_cs [data-testid="stSlider"] [data-testid="stSliderTickBar"], .st-key-mc_cs [data-testid="stSlider"] [data-testid="stSliderTickBar"] * {
  font-size: .64rem !important; background: transparent !important; color: #64748B !important; font-weight: 400 !important; padding: 0 !important; }
.st-key-mc_cs [data-testid="stSlider"] [data-testid="stSliderThumbValue"], .st-key-mc_cs [data-testid="stSlider"] [data-testid="stThumbValue"] { font-size: .85rem; font-weight: 500; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem !important; }
[data-testid="stSidebar"] hr { margin: .6rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"], [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child { padding-top: .5rem !important; }
.st-key-nav_section [data-testid="stWidgetLabel"] { margin-bottom: 0 !important; }
.st-key-nav_section [role="radiogroup"] { gap: .1rem; }
.st-key-nav_section [role="radiogroup"] > div { width: 100%; }
.st-key-nav_section label[data-testid="stRadioOption"], .st-key-nav_section label[data-baseweb="radio"] { display: flex !important; box-sizing: border-box; }
.st-key-nav_section label[data-testid="stRadioOption"], .st-key-nav_section label[data-baseweb="radio"] { width: 100%; padding: .2rem .7rem !important; border: 1px solid transparent; border-radius: 8px; background: transparent; cursor: pointer; min-height: 0 !important; }
.st-key-nav_section label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none !important; }
.st-key-nav_section label[data-baseweb="radio"] > div:first-child:not(:has([data-testid="stMarkdownContainer"])) { display: none !important; }
.st-key-nav_section label[data-testid="stRadioOption"]:hover, .st-key-nav_section label[data-baseweb="radio"]:hover { background: #F0FDF4; }
.st-key-nav_section label[data-testid="stRadioOption"]:has(input:checked), .st-key-nav_section label[data-baseweb="radio"]:has(input:checked) { background: #DCFCE7 !important; border-color: #16A34A; }
.st-key-nav_section label[data-testid="stRadioOption"] p, .st-key-nav_section label[data-baseweb="radio"] p { color: #111827 !important; font-weight: 400 !important; }
.st-key-mc_cs [data-testid="stWidgetLabel"] p { font-weight: 600 !important; font-size: .9rem !important; color: #334155 !important; }
</style>""", unsafe_allow_html=True)


def render_sidebar(show_gk: bool) -> None:
    """One input section is shown at a time; picking another section replaces it."""
    sections = ["Total corpus", "Spend", "Returns and inflation", "Asset allocation", "Risk protection"]
    if show_gk:
        sections.append("Guardrails (Guyton–Klinger)")
    if st.session_state.get("nav_section") not in sections:
        st.session_state["nav_section"] = sections[0]

    with st.sidebar:
        st.markdown("**Inputs: pick a section**")
        def section_label(name: str) -> str:
            """Show the corpus and the total first-year spend (essential + discretionary) in crore next to their buttons."""
            def cr(x: float) -> str:
                t = f"{x:,.3f}"
                return "₹" + (t[:-1] if t.endswith("0") else t) + " Cr"
            ss_ = st.session_state
            if name == "Total corpus":
                return f"{name} (**{cr(float(ss_['corpus_cr']))}**)"
            if name == "Spend":
                total_l = float(ss_['essential_l']) + float(ss_['discretionary_l'])
                if total_l < 100:   # below 1 Cr: show in lakh
                    t = f"{total_l:,.2f}".rstrip("0").rstrip(".")
                    return f"{name} (**₹{t} L**)"
                return f"{name} (**{cr(total_l / 100.0)}**)"
            return name
        _mc_slider_css()
        choice = st.radio("Section", sections, key="nav_section", format_func=section_label, label_visibility="collapsed")
        st.divider()

        if choice == "Total corpus":
            _mc_slider_css()
            with st.container(key="mc_cs"):
                _exact_slider("Starting corpus", "corpus_cr", CORPUS_OPTS_MC, _fmt_cr_l)
            st.radio("Retirement duration (years)", [30, 40, 50], format_func=lambda years: f"{years}yr", horizontal=True, key="years")
            st.caption(f"Ages in the charts assume retirement starts at age {START_AGE}.")

        elif choice == "Spend":
            _mc_slider_css()
            with st.container(key="mc_cs"):
                _exact_slider("Essential spend (year 1)", "essential_l", SPEND_OPTS_MC, _fmt_l_cr)
                _exact_slider("Discretionary spend (year 1)", "discretionary_l", SPEND_OPTS_MC, _fmt_l_cr)
            st.radio("Separate medical corpus planned?", ["Yes", "No"], horizontal=True, key="medical_separate")
            if st.session_state["medical_separate"] == "No":
                st.number_input(
                    "Medical spend (₹ lakh, first retirement year, incl. insurance premiums)",
                    min_value=0.0, step=0.5, key="medical_l",
                )
                st.slider("Medical inflation premium over general inflation (pp)", 0.0, 10.0, step=0.25,
                          key="medical_premium_pct")
                st.caption(
                    "Medical spending is paid from this corpus, grows faster than general inflation, "
                    "and is never tapered or cut. The ₹3 L and 3 pp are placeholders; replace with your own figures."
                )
            else:
                st.caption("Medical costs are excluded. Enter only the non-medical corpus in the starting corpus field.")
            st.number_input("Taper start age (0 = disabled)", min_value=0, max_value=120, step=1, key="taper_age")
            st.slider("Annual real spending reduction after taper age (%)", 0.0, 20.0, step=0.25, key="taper_pct")
            st.checkbox("Apply taper to essential spending", key="taper_essential")
            st.checkbox("Apply taper to discretionary spending", key="taper_discretionary")

        elif choice == "Returns and inflation":
            st.slider("Expected equity return (arithmetic mean, %)", -20.0, 30.0, step=0.25, key="equity_mean_pct")
            st.slider("Equity volatility (%)", 0.0, 60.0, step=0.5, key="equity_vol_pct")
            st.checkbox("Fat-tailed equity returns (Student-t, 5 df)", key="fat_tails")
            st.caption("Same mean and volatility as the normal option, but crashes and spikes are more frequent.")
            st.slider("Debt return (%)", -10.0, 20.0, step=0.25, key="debt_return_pct")
            st.slider("Expected inflation (mean, %)", -2.0, 20.0, step=0.25, key="inflation_pct")
            st.slider("Inflation volatility (standard deviation, %)", 0.0, 15.0, step=0.25, key="inflation_vol_pct")

        elif choice == "Asset allocation":
            st.slider("Target equity allocation of invested assets (%)", 0, 100, step=5, key="equity_pct")
            st.selectbox("Rebalance invested equity/debt allocation", list(REBALANCE_OPTIONS), key="rebalance_label")
            st.caption(
                "The target applies to invested assets only. The safe bucket is held outside it, "
                "so equity as a share of the whole corpus is lower (shown below the results)."
            )

        elif choice == "Risk protection":
            st.slider("Safe bucket (years of initial total spending)", 0.0, 15.0, step=0.5, key="safe_years")
            st.checkbox("Bucket refill: use bucket in down years, refill after good years", key="bucket_refill")
            st.caption(
                "On: the bucket is drawn only when the invested portfolio falls in a year, and is topped "
                "back up (from invested assets) after other years. Off (default): the bucket is spent first and never refilled. "
                "In testing, refill mode lowered success rates because it keeps years of spending in a low-return asset indefinitely."
            )
            st.slider("Discretionary cut in negative portfolio-return years (%)", 0.0, 100.0, step=1.0, key="cut_pct")
            st.selectbox("Historical initial-market stress", [NONE_LABEL, GFC_LABEL], key="historical_sequence")
            st.caption(
                "When selected, the first five retirement-year equity returns use the historical "
                "Nifty 50 TRI sequence (approximately −51%, +78%, +19%, −24%, +29%). "
                "Monte Carlo returns resume after Year 5."
            )
            if st.session_state["historical_sequence"] == GFC_LABEL:
                st.slider("Extra inflation during those 5 years (pp)", 0.0, 8.0, step=0.5, key="stress_extra_pct")

        elif choice == "Guardrails (Guyton–Klinger)":
            st.checkbox("Apply guardrails", key="gk_on")
            st.slider("Upper guardrail: withdrawal rate above starting rate by (%)", 5, 60, step=1, key="gk_upper_pct")
            st.slider("Cut spending by (%) when upper guardrail is hit", 1, 30, step=1, key="gk_cut_pct")
            st.slider("Lower guardrail: withdrawal rate below starting rate by (%)", 5, 60, step=1, key="gk_lower_pct")
            st.slider("Raise spending by (%) when lower guardrail is hit", 1, 30, step=1, key="gk_raise_pct")
            st.checkbox("Skip the inflation increase after a negative-return year", key="gk_skip")
            st.slider("Spending floor: never spend less than (% of plan, today's ₹)", 30, 100, step=5, key="gk_floor_pct")
            st.caption(
                "Withdrawal rate = planned spending ÷ portfolio at the start of the year. Guardrails apply "
                "to essential and discretionary spending only, and replace the discretionary cut in "
                "negative-return years (Risk protection section). Cuts stop in the final 15 years of the horizon. "
                "The spending floor is a hard limit: cuts never go below it, and if that is not enough "
                "the money runs out."
            )


def main() -> None:
    st.title("Retirement Monte Carlo Calculator")
    st.caption(
        "Success means every scheduled expense is fully funded through the selected horizon. "
        "The chart shows future (nominal) rupees; the headline figures and legacy are in today's rupees. "
        f"Ages assume retirement starts at {START_AGE}. Taxes and fees are not modelled."
    )

    init_state()
    ss = st.session_state
    inputs = build_inputs()

    try:
        result = cached_run(asdict(inputs))
    except ValueError as e:
        render_sidebar(False)
        alert("error", str(e))
        st.stop()

    show_gk = result["success_rate"] * 100.0 < GK_TRIGGER_PCT
    render_sidebar(show_gk)
    legacy_conf = int(ss["legacy_conf"])
    legacy_age_in = int(ss["legacy_age_in"])
    gk_floor_pct = int(ss["gk_floor_pct"])

    st.markdown(KPI_CSS, unsafe_allow_html=True)

    last_age = inputs.start_age + inputs.years - 1
    check_age = 90 if inputs.start_age <= 90 <= last_age else last_age
    success_pct = result["success_rate"] * 100.0
    funded_pct = funded_through_age(result, inputs.start_age, check_age) * 100.0

    terminal_real = result["terminal_real"]
    median_all = float(np.median(terminal_real))

    kpi_values = [
        ("Probability of Success", f"{success_pct:.2f}%",
         f"success-value {status_class(success_pct)}"),
        (f"Funded Through Age {check_age}", f"{funded_pct:.2f}%",
         f"success-value {status_class(funded_pct)}"),
        (
            "Median Corpus at End (today's ₹)",
            f'<span class="retirement-kpi-subvalue {"corpus-positive" if median_all >= 1.0 else "corpus-zero"}">All paths: {fmt_money(median_all)}</span>',
            "median-value",
        ),
        ("Initial Safe Bucket", fmt_money(result["initial_safe"]),
         "safe-positive" if result["initial_safe"] > 0 else "safe-zero"),
    ]
    kpi_cols = st.columns([0.95, 1.0, 1.8, 0.95])
    for idx, (label, value, value_class) in enumerate(kpi_values):
        last_class = " last" if idx == len(kpi_values) - 1 else ""
        label_class = "median-label" if label.startswith("Median") else ""
        with kpi_cols[idx]:
            st.markdown(
                f'<div class="retirement-kpi{last_class}">'
                f'<div class="retirement-kpi-label {label_class}">{label}</div>'
                f'<div class="retirement-kpi-value {value_class}">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    eff_equity = (inputs.equity_pct * result["initial_invested"] / inputs.corpus * 100.0) if inputs.corpus > 0 else 0.0
    withdrawal_rate = (result["first_year_spend"] / inputs.corpus * 100.0) if inputs.corpus > 0 else float("nan")
    alert("info", 
        f"Initial corpus: {fmt_money(inputs.corpus)} | "
        f"Safe bucket: {fmt_money(result['initial_safe'])} | "
        f"Invested: {fmt_money(result['initial_invested'])} | "
        f"Equity as share of whole corpus: {eff_equity:.1f}% | "
        f"First-year spending: {fmt_money(result['first_year_spend'])} ({withdrawal_rate:.2f}% of corpus) | "
        f"Expected invested-portfolio arithmetic return: "
        f"{(inputs.equity_pct*inputs.equity_mean+(1-inputs.equity_pct)*inputs.debt_return)*100:.2f}%"
    )
    if inputs.debt_return < inputs.inflation:
        alert("caution", 
            "Debt return is below expected inflation, so debt and the safe bucket lose purchasing power "
            "by construction. Confirm this is intended."
        )

    # ---------------- Guyton–Klinger guardrails (offered only when success is below the trigger) ----------------
    gk_result = None
    gk_inputs = None
    if show_gk:
        alert("caution", 
            f"Success probability is below {GK_TRIGGER_PCT:.0f}%. Guyton–Klinger guardrails can improve it: "
            "adjust them in the Guardrails section of the sidebar."
        )
        if ss["gk_on"]:
            gk_inputs = replace(
                inputs, gk_enabled=True, gk_upper=ss["gk_upper_pct"] / 100.0, gk_lower=ss["gk_lower_pct"] / 100.0,
                gk_cut=ss["gk_cut_pct"] / 100.0, gk_raise=ss["gk_raise_pct"] / 100.0,
                gk_skip_inflation=bool(ss["gk_skip"]), gk_floor=gk_floor_pct / 100.0,
            )
            try:
                gk_result = cached_run(asdict(gk_inputs))
            except ValueError as e:
                alert("error", str(e))

    failed = ~result["success"]
    if failed.any():
        n_failed = int(failed.sum())
        st.markdown(
            f'<div class="rm-failed-row"><span><strong class="rm-failed-count">{n_failed:,} of {inputs.sims:,}</strong> paths failed '
            f'({n_failed / inputs.sims * 100:.1f}%)</span></div>',
            unsafe_allow_html=True,
        )
    else:
        alert("success", "All simulated paths funded every scheduled withdrawal.")

    # ---------------- Chart 1: real-terms percentile fan ----------------
    nominal = result["nominal_balances"]
    ages = np.arange(inputs.start_age, inputs.start_age + inputs.years + 1)
    p10, p25, p50, p75, p90 = np.percentile(nominal, [10, 25, 50, 75, 90], axis=1)
    median_end_nominal = float(p50[-1])
    median_end_pv = float(np.median(result["terminal_real"]))  # today's rupees

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Sampled individual paths (fixed sample so the picture is stable). Drawn from the 2nd-99th
    # percentile of terminal outcomes, so some lines fall below the 10th and rise above the 90th
    # percentile bands. Percentile bands and median still use every simulation.
    terminal = nominal[-1]
    lo, hi = np.percentile(terminal, [2, 99])
    pool = np.flatnonzero((terminal >= lo) & (terminal <= hi))
    # Leave out sampled paths that ever exceed the chart's y-axis cap, so no line runs off the top.
    pool = pool[nominal[:, pool].max(axis=0) / 1e7 <= Y_CAP_CR]
    if pool.size:
        pick = np.random.default_rng(20261106).choice(pool, size=min(100, pool.size), replace=False)
        for i in pick:
            ax.plot(ages, nominal[:, i] / 1e7, color="#ADD8E6", linewidth=0.6, alpha=0.20, label="_nolegend_")

    ax.fill_between(ages, p10 / 1e7, p90 / 1e7, alpha=0.18, label="10th–90th percentile")
    ax.fill_between(ages, p25 / 1e7, p75 / 1e7, alpha=0.30, label="25th–75th percentile")
    ax.plot(ages, p50 / 1e7, linewidth=2.5, label="Median")
    ax.set_xlabel("Age (balance at start of first year, then after each year's withdrawal)")
    ax.set_ylabel("Portfolio value (₹ crore, Future value)")
    ax.set_title(
        "Portfolio Trajectories (Future value)\n"
        f"Median at age {ages[-1]}: {fmt_money(median_end_nominal) if median_end_nominal < 1e9 else f'₹{median_end_nominal / 1e7:,.0f} Cr'}"
        f" ({fmt_money(median_end_pv)} in today's ₹)"
    )
    # Headroom above the 90th percentile so lines above it are visible, but never above the cap.
    y_max = min(float(np.max(p90 / 1e7)) * 1.5, Y_CAP_CR)
    ax.set_ylim(0, y_max if y_max > 0 else 1)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Future value = nominal rupees of each future year. The y-axis stops at ₹{Y_CAP_CR:,.0f} Cr. "
        "Thin lines are up to 100 sampled paths; any path that goes above that limit is left out of the sample. "
        "Percentile bands and the median use every simulation."
    )

    # ---------------- Chart 2: ruin by age ----------------
    fy = result["failure_year"]
    fail_counts = np.bincount(fy[fy >= 0], minlength=inputs.years)[: inputs.years]
    cum_fail = np.cumsum(fail_counts) / inputs.sims * 100.0
    fail_ages = np.arange(inputs.start_age, inputs.start_age + inputs.years)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        ax2.plot(fail_ages, cum_fail, linewidth=2.2, color="#C62828")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Cumulative probability of funding failure (%)")
        ax2.set_title("Ruin probability by age")
        ax2.set_ylim(0, max(5.0, float(cum_fail.max()) * 1.1))
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
    with col_b:
        st.markdown("**Probability of being fully funded through age**")
        milestones = [a for a in (70, 75, 80, 85, 90, 95, 100, 105) if inputs.start_age <= a <= last_age]
        if last_age not in milestones:
            milestones.append(last_age)
        funded_values = [round(funded_through_age(result, inputs.start_age, a) * 100.0, 1) for a in milestones]
        table = pd.DataFrame({
            "Age": milestones,
            "Funded (%)": funded_values,
            "Status": [status_icon(v) for v in funded_values],
        })
        st.dataframe(table, hide_index=True, width="stretch")

    # ---------------- Legacy: amount left behind at a chosen confidence ----------------
    end_age = inputs.start_age + inputs.years
    if int(legacy_age_in) == 0:
        legacy_age = end_age
    else:
        legacy_age = min(max(int(legacy_age_in), inputs.start_age), end_age)
    legacy_idx = legacy_age - inputs.start_age
    legacy_bal = result["real_balances"][legacy_idx]
    legacy_value = legacy_amount(result["real_balances"], legacy_idx, float(legacy_conf))
    prob_any = float((legacy_bal > 1.0).mean()) * 100.0

    st.subheader("Legacy: what you can count on leaving behind")
    if legacy_value > 1.0:
        alert("success", 
            f"With {legacy_conf}% confidence, you leave **at least {fmt_money(legacy_value)}** "
            f"(today's ₹) at age {legacy_age}."
        )
    else:
        alert("error", 
            f"At {legacy_conf}% confidence the plan leaves **₹0** at age {legacy_age}. "
            f"Only {prob_any:.1f}% of simulated paths leave anything."
        )
    legacy_rows = [
        {"Confidence": f"{c}%", "Leave at least (today's ₹)": fmt_money(legacy_amount(result["real_balances"], legacy_idx, float(c)))}
        for c in (50, 75, 90, 95, 99)
    ]
    st.dataframe(pd.DataFrame(legacy_rows), hide_index=True, width="stretch")

    # ---------------- Guardrails: before / after ----------------
    if gk_result is not None:
        st.subheader("Improving the odds: Guyton–Klinger guardrails")
        ratio = gk_result["spend_ratio"]  # essential + discretionary spending vs plan, today's ₹; rows = years
        gk_pct = gk_result["success_rate"] * 100.0
        min_ratio = np.nanmin(ratio, axis=0)
        hit_floor_pct = float((min_ratio <= gk_floor_pct / 100.0 + 1e-9).mean()) * 100.0
        any_cut_pct = float((gk_result["cut_count"] > 0).mean()) * 100.0
        path_avg = np.nanmean(ratio, axis=0)  # each path's average spending vs plan, over the years it was funded
        median_spend_pct = float(np.median(path_avg)) * 100.0
        bad_spend_pct = float(np.percentile(path_avg, 10)) * 100.0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                kpi_card("Success without guardrails", f"{success_pct:.1f}%",
                         f"success-value {status_class(success_pct)}"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                kpi_card(
                    f"Success with guardrails (spending never below {gk_floor_pct}% of plan)",
                    f'{gk_pct:.1f}%<span class="retirement-kpi-subvalue gk-pp" style="color:#252936;">{gk_pct - success_pct:+.1f} pp vs without</span>',
                    f"success-value {status_class(gk_pct)}", last=True,
                ),
                unsafe_allow_html=True,
            )
        m3, m4 = st.columns(2)
        m3.metric("Paths with at least one spending cut", f"{any_cut_pct:.1f}%")
        m4.metric("Paths that reach your spending floor", f"{hit_floor_pct:.1f}%")
        m6, m7 = st.columns(2)
        m6.metric("Median spending vs plan (today's ₹)", f"{median_spend_pct:.1f}%")
        m7.metric("Bad-case spending vs plan (10th percentile)", f"{bad_spend_pct:.1f}%")
        st.caption(
            "A path is one simulated future. Guardrails are checked once a year, so a path has at most one cut "
            "in any year, but it can be cut in several different years. Median spending vs plan is the middle "
            "path's average over the years it was funded. A path counts as funded only if every year's spending, "
            "held at or above your floor, was paid in full."
        )

        # ---- Floor check: the highest floor that still reaches the green success level ----
        if gk_inputs is not None:
            scan = floor_scan(gk_inputs)
            scan_map = dict(scan)
            s_user = scan_map.get(gk_floor_pct, gk_pct)
            ok_floors = [f for f, sc in scan if sc >= GREEN_AT]
            best = max(ok_floors) if ok_floors else None
            if best is None:
                top_f, top_s = max(scan, key=lambda t: t[1])
                alert("warning", 
                    f"**Floor check:** no spending floor from 30% to 100% reaches {GREEN_AT:.0f}% success with your "
                    f"guardrail settings (best is {top_s:.1f}% at a {top_f}% floor). Guardrails alone cannot fix this "
                    "plan: it needs a larger corpus, lower planned spending or a shorter horizon."
                )
            elif s_user < GREEN_AT:
                alert("warning", 
                    f"**Floor check:** your {gk_floor_pct}% floor looks too optimistic. Success is about {s_user:.1f}%, "
                    f"below the {GREEN_AT:.0f}% green level. Suggested floor: **{best}%** "
                    f"(about {scan_map[best]:.1f}% success). Expect deeper cuts in bad futures."
                )
            elif best >= gk_floor_pct + 5:
                alert("info", 
                    f"**Floor check:** your {gk_floor_pct}% floor is lower than it needs to be. Success stays at or "
                    f"above {GREEN_AT:.0f}% with a floor as high as **{best}%** (about {scan_map[best]:.1f}% success). "
                    f"Raising your floor to {best}% protects more of your spending."
                )
            else:
                alert("success", 
                    f"**Floor check:** your {gk_floor_pct}% floor is about right. The highest floor that still "
                    f"reaches {GREEN_AT:.0f}% success is {best}% (about {scan_map[best]:.1f}% success)."
                )
            with st.expander("Success at each spending floor"):
                st.dataframe(
                    pd.DataFrame({
                        "Floor (% of plan)": [f for f, _ in scan],
                        "Success (%)": [round(sc, 1) for _, sc in scan],
                        "Status": [status_icon(sc) for _, sc in scan],
                    }),
                    hide_index=True, width="stretch",
                )
                st.caption(
                    "Each floor is tested with your other guardrail settings unchanged, using up to 10,000 "
                    "simulations, so figures can differ slightly from the headline."
                )

        g_a, g_b = st.columns(2)
        with g_a:
            fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
            ax3.plot(fail_ages, cum_fail, linewidth=2.2, color="#C62828", label="Without guardrails")
            gk_fy = gk_result["failure_year"]
            gk_counts = np.bincount(gk_fy[gk_fy >= 0], minlength=inputs.years)[: inputs.years]
            gk_cum = np.cumsum(gk_counts) / inputs.sims * 100.0
            ax3.plot(fail_ages, gk_cum, linewidth=2.2, color="#16803C", label="With guardrails")
            ax3.set_xlabel("Age")
            ax3.set_ylabel("Cumulative probability of funding failure (%)")
            ax3.set_title("Ruin probability by age")
            ax3.set_ylim(0, max(5.0, float(cum_fail.max()) * 1.1))
            ax3.grid(alpha=0.3)
            ax3.legend()
            st.pyplot(fig3)
            plt.close(fig3)
        with g_b:
            counts = np.sum(~np.isnan(ratio), axis=1)
            enough = counts >= max(30, int(0.01 * inputs.sims))
            q10 = np.full(inputs.years, np.nan)
            q50 = np.full(inputs.years, np.nan)
            q90 = np.full(inputs.years, np.nan)
            for t in np.flatnonzero(enough):
                q10[t], q50[t], q90[t] = np.nanpercentile(ratio[t], [10, 50, 90])
            fig4, ax4 = plt.subplots(figsize=(6.5, 4.2))
            ax4.fill_between(fail_ages, q10 * 100, q90 * 100, alpha=0.20, color="#185AAB", label="10th–90th percentile")
            ax4.plot(fail_ages, q50 * 100, linewidth=2.2, color="#185AAB", label="Median")
            ax4.axhline(100, color="grey", linewidth=1, linestyle="--", label="Your plan")
            ax4.axhline(gk_floor_pct, color="#C62828", linewidth=1.8, linestyle="-", label="Your spending floor")
            ax4.set_xlabel("Age")
            ax4.set_ylabel("Spending vs plan (%, today's ₹)")
            ax4.set_title("Spending under guardrails (paths still funded)")
            ax4.grid(alpha=0.3)
            ax4.legend()
            st.pyplot(fig4)
            plt.close(fig4)

    # ---------------- Sensitivity ----------------
    with st.expander("Sensitivity: how fragile is this answer?"):
        st.caption(
            "One assumption changed at a time, same random seed, up to 10,000 simulations per row. "
            "Large swings mean the headline number depends heavily on that assumption."
        )
        if st.checkbox("Run sensitivity table"):
            st.dataframe(sensitivity_table(inputs), hide_index=True, width="stretch")

    st.subheader("How to read this page")
    topics = explanation_topics(inputs, result, ss)
    chosen = st.selectbox("Pick an item to see what it means", list(topics.keys()), key="explain_topic")
    st.markdown(topics[chosen])



# ======================================================================
# STRESS TEST: page
# ======================================================================

def run_stress_test() -> None:
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --page: #FAFBFD;
  --sidebar: #F3F5F9;
  --ink: #1E293B;      /* headings, key figures */
  --body: #3B4A5E;     /* running text */
  --muted: #6B7A90;    /* captions, secondary labels */
  --line: #E2E7EF;
  --accent: #4F6F9F;
  --accent-soft: #F1F5FA;
}
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif; }
.stApp { background: var(--page); color: var(--body); }
[data-testid="stMain"] { background: var(--page); }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.4rem; }
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { background: var(--sidebar); }

/* Body text: one calm size and colour */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th { font-size: .95rem; line-height: 1.65; color: var(--body); }
[data-testid="stMarkdownContainer"] strong { font-weight: 600; color: var(--ink); }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; margin: .6rem 0 1rem; }
[data-testid="stMarkdownContainer"] th { background: #F1F4F9; text-align: left; font-weight: 600; color: var(--ink); font-size: .88rem; }
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th {
  padding: .6rem .8rem; border: 1px solid var(--line); vertical-align: top; }

/* Headings: clear hierarchy, moderate weight */
[data-testid="stHeading"] h1, [data-testid="stMarkdownContainer"] h1 {
  font-size: 1.95rem !important; font-weight: 700 !important; letter-spacing: -.02em !important; color: var(--ink) !important; padding-bottom: .1rem; }
[data-testid="stHeading"] h2, [data-testid="stMarkdownContainer"] h2 {
  font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -.015em; color: var(--ink) !important; }
[data-testid="stHeading"] h3, [data-testid="stMarkdownContainer"] h3 {
  font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -.01em; color: var(--ink) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: .84rem !important; line-height: 1.55; }

/* Slider min/max labels: 25% smaller, no highlight */
[data-testid="stSlider"] [data-testid="stSliderTickBar"],
[data-testid="stSlider"] [data-testid="stSliderTickBar"] * {
  font-size: .64rem !important; line-height: 1.2 !important; background: transparent !important; background-color: transparent !important;
  color: var(--muted) !important; font-weight: 400 !important; padding: 0 !important; box-shadow: none !important; }

/* Info / warning boxes */
[data-testid="stAlert"] p { font-size: .92rem; line-height: 1.6; }

/* Sidebar: quiet labels, readable captions */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
  color: #334155 !important; font-weight: 600 !important; font-size: .9rem !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom: .25rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: .8rem !important; line-height: 1.5; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stThumbValue"] { font-size: .85rem; font-weight: 500; }
[data-testid="stRadio"] label p { font-weight: 500 !important; font-size: .88rem !important; color: #40516A !important; }

/* Expanders */
[data-testid="stExpander"] {
  background: #FFFFFF !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
  overflow: hidden; margin: .5rem 0 .8rem; box-shadow: none; }
[data-testid="stExpander"] details > summary {
  background: var(--accent-soft) !important; padding: .65rem .9rem !important; transition: background-color .15s ease; }
[data-testid="stExpander"] details > summary:hover { background: #E9EFF7 !important; }
[data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid var(--line); }
[data-testid="stExpander"] details > summary p,
[data-testid="stExpander"] details > summary span { color: #2F4360 !important; font-weight: 600 !important; font-size: .92rem !important; }

/* Dropdowns: white with a quiet border */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; min-height: 2.5rem; font-size: .92rem; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover { border-color: #91A9C7 !important; }
/* Slider track: red up to the selected value */
[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background-color: #E53935 !important;
}

/* Slider thumb: red */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #E53935 !important;
    border-color: #E53935 !important;
}

/* Unselected portion of the slider: light grey */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-color: #DCE2EC;
}
[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child { filter: hue-rotate(222deg) saturate(.62) brightness(1.02); }
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child:not(:has([data-testid="stMarkdownContainer"])) { filter: hue-rotate(222deg) saturate(.62) brightness(1.02); }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="input"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #FFFFFF !important; }
/* Segmented toggles for Country and Withdrawal method (works with older and newer Streamlit markup) */
.st-key-country, .st-key-method, .st-key-country [data-testid="stRadio"], .st-key-method [data-testid="stRadio"] { width: 100% !important; }
.st-key-country [data-testid="stRadio"] > div:last-child, .st-key-method [data-testid="stRadio"] > div:last-child { width: 100% !important; }
.st-key-country [role="radiogroup"] {
  display: flex !important; flex-wrap: nowrap !important; gap: 2px; background: #E6EBF3; padding: 2px; border-radius: 9px; width: 100%; }
.st-key-country [role="radiogroup"] > div, .st-key-country [role="radiogroup"] > label { flex: 1 1 0; min-width: 0; }
.st-key-country [role="radiogroup"] label {
  width: 100%; box-sizing: border-box; justify-content: center; margin: 0 !important; padding: .22rem .1rem; border-radius: 7px; position: relative; gap: 0 !important;
  cursor: pointer; transition: background-color .15s ease, box-shadow .15s ease; }
.st-key-country [role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
.st-key-country [role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none !important; }
.st-key-country [role="radiogroup"] label > div:last-child { justify-content: center; width: 100%; }
.st-key-country [role="radiogroup"] label > span, .st-key-method [role="radiogroup"] label > span { position: absolute; opacity: 0; width: 0; height: 0; overflow: hidden; }
.st-key-country [role="radiogroup"] label p {
  font-size: .975rem !important; font-weight: 400 !important; color: #475569 !important; text-align: center; margin: 0; white-space: nowrap; }
.st-key-country [role="radiogroup"] label:hover { background: rgba(255,255,255,.7); }
.st-key-country [role="radiogroup"] label:has(input:checked) {
  background: #DCFCE7; box-shadow: 0 0 0 1px #16A34A inset; }
.st-key-country [role="radiogroup"] label:has(input:checked) p { color: #111827 !important; font-weight: 400 !important; }
.st-key-method [role="radiogroup"] {
  display: flex !important; flex-wrap: nowrap !important; gap: 2px; background: #E6EBF3; padding: 2px; border-radius: 9px; width: 100%; }
.st-key-method [role="radiogroup"] > div, .st-key-method [role="radiogroup"] > label { flex: 1 1 auto; min-width: 0; }
.st-key-method [role="radiogroup"] label {
  width: 100%; box-sizing: border-box; justify-content: center; margin: 0 !important; padding: .22rem .1rem; border-radius: 7px; position: relative; gap: 0 !important;
  cursor: pointer; transition: background-color .15s ease, box-shadow .15s ease; }
.st-key-method [role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
.st-key-method [role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none !important; }
.st-key-method [role="radiogroup"] label > div:last-child { justify-content: center; width: 100%; }
.st-key-method [role="radiogroup"] label p {
  font-size: .975rem !important; font-weight: 400 !important; color: #475569 !important; text-align: center; margin: 0; white-space: nowrap; }
.st-key-method [role="radiogroup"] label:hover { background: rgba(255,255,255,.55); }
.st-key-method [role="radiogroup"] label:has(input:checked) {
  background: #DCFCE7; box-shadow: 0 0 0 1px #16A34A inset; }
.st-key-method [role="radiogroup"] label:has(input:checked) p { color: #111827 !important; font-weight: 400 !important; }
.rule-events { max-height: 420px; overflow-y: auto; border: 1px solid var(--line); border-radius: 8px; background: #FFFFFF; }
.rule-events table { width: 100%; border-collapse: collapse; margin: 0; table-layout: auto; }
.rule-events th, .rule-events td { padding: .5rem .5rem; border: none; border-bottom: 1px solid var(--line);
  font-size: .88rem; line-height: 1.4; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: anywhere; }
.rule-events th { position: sticky; top: 0; background: #F3F6FA; color: var(--muted); font-weight: 600; z-index: 1; }
.rule-events th:nth-child(1), .rule-events td:nth-child(1) { width: 1%; white-space: nowrap; }
.rule-events th:nth-child(3), .rule-events td:nth-child(3),
.rule-events th:nth-child(4), .rule-events td:nth-child(4) { width: 1%; white-space: nowrap; }
.rule-events tr:last-child td { border-bottom: none; }
</style>""", unsafe_allow_html=True)
    st.title("Historical Stress Test")
    st.caption("Replays real market and inflation history. No simulation. Success = corpus above zero after the last year. Real values are expressed Inflation Adjusted.")

    with st.sidebar:
        country = st.radio("Country", ["India", "USA", "Japan"], index=0, horizontal=True, key="country")
        path_caption = {
            "USA": "1929 · Great Depression  |  1966 · Inflation and rising rates",
            "India": "1986 · High inflation, weak returns  |  2000 · Dot-com bust",
            "Japan": "1990 · Bubble collapse  |  2000 · Dot-com downturn and deflation",
        }
        st.caption(path_caption[country])
        method_choice = st.radio("Withdrawal method", ["Fixed", "Guyton-Klinger"], horizontal=True, key="method")
        strategy = "Guyton-Klinger" if method_choice == "Guyton-Klinger" else "Fixed withdrawals (inflation-adjusted)"
        st.caption("Inflation-adjusted withdrawals every year." if method_choice == "Fixed"
                   else "Spending flexes with the portfolio (guardrails).")
        # The chosen values live in session_state in rupees (set once at app start). A widget per country is filled
        # from them before it is drawn, so switching country keeps your selection, and each country's slider can use
        # its own steps (rupee steps for India, round dollar steps for USA).
        ss = st.session_state
        ss.setdefault("st_corpus_cr", 5.0)
        ss.setdefault("st_spend_cr", 0.24)
        if country == "USA":
            usd_cr = lambda m: m * 1e6 * USD_INR / 1e7          # $ million -> crore
            usd_sp = lambda k: k * 1e3 * USD_INR / 1e7          # $ thousand -> crore
            ss["corpus_w_USA"] = nearest(USD_CORPUS_OPTIONS, usd_cr, ss["st_corpus_cr"])
            ss["spend_w_USA"] = nearest(USD_SPEND_OPTIONS, usd_sp, ss["st_spend_cr"])
            corpus_usd = st.select_slider("Starting corpus", options=USD_CORPUS_OPTIONS,
                                          format_func=lambda m: f"${m:.1f} M ({money(usd_cr(m))})", key="corpus_w_USA",
                                          on_change=lambda: ss.update(st_corpus_cr=usd_cr(ss["corpus_w_USA"])))
            spend_usd = st.select_slider("First-year expenses (per year)", options=USD_SPEND_OPTIONS,
                                         format_func=lambda k: f"${k} K ({money(usd_sp(k))})", key="spend_w_USA",
                                         on_change=lambda: ss.update(st_spend_cr=usd_sp(ss["spend_w_USA"])))
            corpus, spend_lakh = usd_cr(corpus_usd), usd_sp(spend_usd) * 100
        else:
            country_key = "India" if country == "India" else "Japan"
            ss[f"corpus_w_{country_key}"] = nearest(CORPUS_OPTIONS, lambda o: o, ss["st_corpus_cr"])
            ss[f"spend_w_{country_key}"] = nearest(SPEND_OPTIONS, lambda o: o / 100, ss["st_spend_cr"])
            corpus = st.select_slider("Starting corpus", options=CORPUS_OPTIONS,
                                      format_func=lambda v: f"₹{v:g} Cr", key=f"corpus_w_{country_key}",
                                      on_change=lambda: ss.update(st_corpus_cr=ss[f"corpus_w_{country_key}"]))
            spend_lakh = st.select_slider("First-year expenses (per year)", options=SPEND_OPTIONS,
                                          format_func=lambda v: f"₹{v} L" if v < 100 else f"₹{v / 100:g} Cr", key=f"spend_w_{country_key}",
                                          on_change=lambda: ss.update(st_spend_cr=ss[f"spend_w_{country_key}"] / 100))
        spend = spend_lakh / 100  # in crore
        taper = st.slider("Taper after age 80 (% per year)", 0.0, 5.0, 2.0, 0.5, key="st_taper") / 100
        age0 = START_AGE
        equity = st.slider("Equity share (%)", 0, 100, 60, 5, key="st_equity") / 100
        india_debt_options = {
            "Matches Inflation": ("Matches Inflation", 0.0),
            "-1%": ("Inflation - 1%", -0.01),
            "-2%": ("Inflation - 2%", -0.02),
        }
        india_debt_selection = "Matches Inflation"
        india_debt_method_label = "Matches Inflation"
        india_debt_real_assumption = 0.0
        if country == "India":
            st.markdown("**Debt Returns**")
            india_debt_selection = st.radio(
                "Debt Returns",
                list(india_debt_options),
                key="st_debt",
                horizontal=True,
                label_visibility="collapsed",
                help="Sets India's annual debt return relative to that financial year's CPI. This is a sensitivity assumption, not a tax calculation."
            )
            india_debt_method_label, india_debt_real_assumption = india_debt_options[india_debt_selection]
        st.markdown("**Taxation**")
        tax_rate = st.radio(
            "Taxation",
            [0.10, 0.15, 0.20, 0.25, 0.30],
            index=0,
            key="st_tax",
            format_func=lambda v: f"{v:.0%}",
            horizontal=True,
            label_visibility="collapsed",
            help="Tax is applied only to the annual withdrawal. The corpus pays the planned withdrawal plus this tax."
        )
        st.caption(f"Tax on withdrawals: {tax_rate:.0%}. The corpus is debited by the withdrawal plus tax.")
        st.caption(f"{money(corpus, country)} corpus, {money(spend, country)} net expenses: initial withdrawal rate {spend / corpus:.2%} before tax. Year-1 corpus debit including tax: {money(spend * (1 + tax_rate), country)}.")
        equity_source = {"USA": "S&P 500 total return", "India": "Sensex TRI", "Japan": "TOPIX price return plus dividends (2021–25 dividends estimated)"}[country]
        debt_source = ("Debt earns the 3-month US T-bill rate. " if country == "USA" else
                       f"India debt earns {india_debt_method_label.lower()} on a compounded basis. " if country == "India" else
                       "Debt earns the Japan annual cash-yield series. ")
        st.caption(f"Equity earns {equity_source}. " + debt_source + "Rebalanced yearly.")
        if country == "USA":
            st.caption("US history is replayed with US dollar returns and US inflation. Amounts are shown in ₹ for "
                       "convenience, with no exchange-rate conversion. Only the withdrawal rate drives the result.")
        elif country == "Japan":
            st.caption("Japan equity returns are TOPIX price returns plus a dividend return (JST estimates to 2020, a 2.3% estimate for 2021–25), not official TOPIX TR. CPI and cash yields come from the supplied Japan workbook. Amounts are displayed in ₹ as model units; results are driven by return, inflation and withdrawal-rate sequences.")

    kw = dict(corpus=corpus, spend=spend, taper=taper, age0=age0, equity=equity,
              india_debt_real_assumption=india_debt_real_assumption, tax_rate=tax_rate)
    fixed = [simulate(country, s, n, mode="fixed", **kw) for s, n in SCENARIOS[country]]
    gk = [simulate(country, s, n, mode="gk", **kw) for s, n in SCENARIOS[country]]
    use_gk = strategy == "Guyton-Klinger"
    main = gk if use_gk else fixed
    pct = round(100 * sum(r["ok"] for r in main) / len(main))
    fixed_pct = round(100 * sum(r["ok"] for r in fixed) / len(fixed))

    m1, m2 = st.columns([2, 3])
    m1.markdown(f"<div style='font-size:1.05rem;font-weight:600;line-height:1.2;color:#475569;letter-spacing:.01em'>Success</div>"
                f"<div style='font-size:2.3rem;font-weight:700;line-height:1.15;color:{'#16a34a' if pct > 0 else '#dc2626'}'>{pct}%</div>",
                unsafe_allow_html=True)

    def show_cards(results, gk_view=False):
        cols = st.columns(len(results))
        for k, (col, r) in enumerate(zip(cols, results)):
            with col:
                st.markdown(f"<div style='font-size:1.45rem;font-weight:700;line-height:1.3;margin:.9rem 0 .15rem;color:{PALETTE[k]}'>"
                            f"Start {slab(country, r['start'])} <span style='font-weight:500;font-size:.95rem;color:#64748B'>&nbsp;· {r['years']} Years</span></div>",
                            unsafe_allow_html=True)
                context = scenario_context(country, r["start"])
                if context:
                    st.markdown(f"<div style='font-size:.86rem;color:#64748B;margin-bottom:.45rem'>{context}</div>",
                                unsafe_allow_html=True)
                st.markdown("✅ **Lasts**" if r["ok"] else f"❌ **Runs out in {ylab(country, r['fail_year'])}**")
                st.write(f"Years funded: {r['years_funded']} of {r['years']}  ·  "
                         + (f"Ends at age {age0 + r['years'] - 1}" if r["ok"] else f"Runs out at age {age0 + r['years_funded']}"))
                fin_col = "#16a34a" if r["ok"] and r["end_nominal"] > 0 else "#dc2626"
                fin_txt = (f"{money(r['end_nominal'], country)} (PV {money(r['end_real'], country)})"
                           if r["ok"] else "₹0 (PV ₹0)")
                st.markdown(f"<div style='font-size:.98rem;color:#3B4A5E'>Final Corpus: "
                            f"<span style='color:{fin_col};font-weight:600'>{fin_txt}</span></div>", unsafe_allow_html=True)
                if gk_view:
                    st.write(f"Average spending vs plan: **{money(r['avg_spend'], country)}** vs {money(r['avg_plan'], country)} a year "
                             f"(**{r['avg_ratio']:.0%}**)")
                    st.write(f"Lowest year: **{money(r['min_spend'], country)}** vs {money(r['min_plan'], country)} planned "
                             f"(**{r['min_ratio']:.0%}**). Amounts in start-year money.")
                    st.write(f"10% cuts: {r['cuts']}  ·  inflation raises skipped: {r['skips']}")
                with st.expander("Year-by-year"):
                    cols_ = ["Year", "Age", "Withdrawn", "Tax", "TotalDebit", "MarketReturn", "DebtReturn", "Return", "Inflation", "EndBalance", "Real"]
                    d = r["df"][cols_].copy()
                    d["Year"] = d["Year"].map(lambda yy: ylab(country, yy))
                    d["MarketReturn"] = (d["MarketReturn"] * 100).round(1)
                    d["DebtReturn"] = (d["DebtReturn"] * 100).round(1)
                    d["Return"] = (d["Return"] * 100).round(1)
                    d["Inflation"] = (d["Inflation"] * 100).round(1)
                    d["Withdrawn"] = (d["Withdrawn"] * 100).round(1)
                    d["Tax"] = (d["Tax"] * 100).round(1)
                    d["TotalDebit"] = d["TotalDebit"].round(3)
                    d["EndBalance"] = d["EndBalance"].round(2)
                    d["Real"] = d["Real"].round(2)
                    equity_return_label = {"USA": "S&P 500", "India": "Sensex TRI", "Japan": "TOPIX + dividends"}[country]
                    debt_return_label = {"USA": "3-mo T-bill", "India": "Debt", "Japan": "Japan cash"}[country]
                    st.dataframe(d.rename(columns={"MarketReturn": f"{equity_return_label} return %",
                                                   "DebtReturn": f"{debt_return_label} return %",
                                                   "Return": "Portfolio return %", "Inflation": "Inflation %",
                                                   "Withdrawn": "Net withdrawal (₹ L)", "Tax": "Tax (₹ L)", "TotalDebit": "Total corpus debit (₹ Cr)", "EndBalance": "End balance - Nominal (₹ Cr)", "Real": "End balance - Inf Adjusted (₹ Cr)"}),
                                 hide_index=True)

    show_cards(main, gk_view=use_gk)
    st.subheader("Portfolio value (₹ Crore, inflation-adjusted)")
    horizons = " and ".join(f"{n} years (start {slab(country, s)})" for s, n in SCENARIOS[country])
    st.info(f"Retirement is assumed to start at age {START_AGE}. Paths tested: {horizons}.")
    draw_chart(main, corpus, country)
    ctx = {n: rolling(country, n, **kw) for n in sorted({n for _, n in SCENARIOS[country]})}
    st.caption("Context, not the headline: fixed withdrawals started in every possible year with the same horizon: "
               + "; ".join(f"{n}-yr paths: {s} of {t} last" for n, (s, t) in ctx.items()) + ".")

    if use_gk or pct < SUCCESS_BAR:  # Guyton-Klinger is explained only when it is relevant
        st.divider()
        st.header("Guyton-Klinger Strategy")
        if not use_gk:
            gk_pct = round(100 * sum(r["ok"] for r in gk) / len(gk))
            if pct < SUCCESS_BAR:
                st.warning(f"Fixed withdrawals succeed in {pct}% of paths, below {SUCCESS_BAR}%. "
                           f"With Guyton-Klinger the result is **{gk_pct}%**.")
                show_cards(gk, gk_view=True)
                draw_chart(gk, corpus, country)
        st.subheader("How it works with your numbers")
        iwr = spend / corpus
        M = lambda v: money(v, country)
        infl_ex = 0.03
        st.markdown(f"""
Guyton-Klinger (GK) lets spending flex with the portfolio instead of staying fixed in real terms. Everything below uses your sidebar inputs and updates when you move a slider.

**Starting point.** You withdraw **{M(spend)}** in year 1 from a **{M(corpus)}** corpus. That is an **initial withdrawal rate of {iwr:.2%}**, the anchor for every rule. Each year the withdrawal rate is that year's planned withdrawal divided by the portfolio at the start of the year.

| Rule | Trigger | At today's planned {M(spend)} it triggers when the portfolio is | Action |
|---|---|---|---|
| **1. Inflation rule** | Last year's portfolio return was negative **and** the withdrawal rate would be above {iwr:.2%} | in a down year with a weak portfolio | Skip the inflation raise. With {infl_ex:.0%} inflation, {M(spend)} would have become {M(spend * (1 + infl_ex))}; it stays at {M(spend)} |
| **2. Capital preservation (cut)** | Withdrawal rate above **{iwr * 1.2:.2%}** (120% of {iwr:.2%}) | below **{M(corpus / 1.2)}** | Cut spending 10%: {M(spend)} becomes {M(spend * 0.9)}. Not applied in the last 15 years of a path |
| **3. Prosperity (raise)** | Withdrawal rate below **{iwr * 0.8:.2%}** (80% of {iwr:.2%}) | above **{M(corpus / 0.8)}** | Raise spending 10%: {M(spend)} becomes {M(spend * 1.1)} |

The portfolio levels scale up with inflation as your planned spending grows. After age 80, the taper ({taper:.1%} a year) is applied to spending first, then the rules above are checked. The price of GK is variable spending: it survives more often, but you must accept cuts in bad stretches.
""")
        st.subheader("What GK actually did on each historical path")
        # Keep the historical paths side by side. Streamlit allocates each
        # column from the available page width and adapts the layout on narrow screens.
        gk_cols = st.columns(len(gk), gap="large")
        for gi, (col, r) in enumerate(zip(gk_cols, gk)):
            with col:
                st.markdown(f"<span style='color:{PALETTE[gi]};font-weight:600;font-size:.95rem'>{slab(country, r['start'])} Start:</span> {'lasts' if r['ok'] else 'runs out in ' + ylab(country, r['fail_year'])}. "
                            f"Spending averaged **{M(r['avg_spend'])}** a year against a plan of {M(r['avg_plan'])} (**{r['avg_ratio']:.0%}**); "
                            f"lowest year **{M(r['min_spend'])}** against {M(r['min_plan'])} (**{r['min_ratio']:.0%}**), in start-year money. "
                            f"{r['cuts']} cuts, {r['skips']} skipped inflation raises.", unsafe_allow_html=True)
                if r["events"]:
                    with st.expander(f"Rule events, {slab(country, r['start'])} Start ({len(r['events'])})"):
                        ev = pd.DataFrame(r["events"], columns=["Year", "Rule", "Before", "After"])
                        ev["Year"] = ev["Year"].map(lambda yy: ylab(country, yy))
                        ev["Before"] = ev["Before"].map(M)
                        ev["After"] = ev["After"].map(M)
                        # Render as an HTML table so each row grows to fit wrapped text (st.dataframe cannot wrap cells).
                        rows_html = "".join(
                            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                            for row in ev.itertuples(index=False)
                        )
                        head_html = "".join(f"<th>{html.escape(c)}</th>" for c in ev.columns)
                        st.markdown(
                            "<div class='rule-events'><table><thead><tr>" + head_html + "</tr></thead>"
                            "<tbody>" + rows_html + "</tbody></table></div>",
                            unsafe_allow_html=True,
                        )

    india_debt_method = (
        f"- India debt selection: **{india_debt_method_label}**. For each India financial year, debt return = (1 + that FY's CPI) x (1 + selected real-return assumption) - 1. The selected real-return assumption is {india_debt_real_assumption:+.0%}.\n"
        "- The India choices are Matches Inflation (0% real), -1% (inflation - 1%), and -2% (inflation - 2%). This selection replaces historical RBI debt yields in India simulations; Sensex returns and CPI are unchanged. The negative-real-return choices are sensitivity assumptions that can illustrate tax and other leakage, not actual tax calculations; withdrawal tax is controlled separately by the Taxation selector.\n"
    ) if country == "India" else ""

    if country == "USA":
        debt_method_text = (
            "- Equity earns S&P 500 total returns. Debt earns the historical 3-month US T-bill rate.\n"
            "- USA debt: Damodaran's 3-month T-bill rate, the annual average of the daily rate (the same series as FRED). The 1928-1933 values come from his earlier data. The US debt series and all US calculations are unchanged.\n"
        )
        country_data_method = (
            "- USA inflation: annual US CPI-U inflation paired with the calendar-year market returns.\n"
        )
    elif country == "India":
        debt_method_text = (
            "- Equity earns Sensex total returns. India debt earns the selected CPI-linked return assumption.\n"
            + india_debt_method
        )
        country_data_method = (
            f"- India equity: financial-year (April-March) Sensex returns, keyed by the FY start year (2019 = FY2019-20, April 2019 to March 2020, which includes the March 2020 crash). FY1986-87 to FY1996-97 use the published price return plus an estimated {DIV_UPLIFT * 100:.2f}% dividend uplift, because the paper's TRI only starts at March 1997. FY1997-98 to FY2024-25 use the published Sensex TRI. FY2025-26 (-6.01%) is not in the paper and was supplied separately.\n"
            "- India inflation: the existing World Bank calendar-year CPI series is retained and converted to April-March years using the same 75:25 weighting (FY2019-20 = 0.75 x 2019 + 0.25 x 2020). Only the year labels/keys have changed; CPI values and methodology are unchanged. Monthly official all-India Combined CPI can be used as a separate validation from 2011 onward, subject to its base-year transition; it has not been substituted into this historical series. FY2025-26 reuses the 2025 rate because 2026 is not yet published."
        )
    else:
        debt_method_text = (
            "- Japan equity earns TOPIX price return plus a dividend return; debt earns the Japan annual cash-yield series.\n"
        )
        country_data_method = (
            "- Japan equity: annual calendar-year TOPIX price return (year-end to year-end, from the supplied workbook; the 1989 and 2025 year-end levels were checked against Bloomberg) plus a dividend return. For 1989-2020 the dividend return is the JST Macrohistory Database Release 6 eq_div_rtn (an independent academic estimate, roughly 0.4% at the 1989 peak rising to about 2.4% by 2020). For 2021-2025 it is a flat 2.3% estimate. This is not the official TOPIX Total Return Index.\n"
            "- The JST eq_tr equity series was deliberately not used: it differs materially from TOPIX year by year (for example 1990: JST -13% against TOPIX -40%), so TOPIX is used as the market benchmark.\n"
            "- Japan CPI inflation (World Bank via FRED) and annual cash yields (FRED annual-average T-bill yield) are taken from the supplied Japan 1989-Today workbook, giving aligned equity/CPI/cash observations for 1989-2025."
        )

    # Country-specific historical stress periods. Returns are taken directly from
    # the embedded annual total-return series; event descriptions provide context.
    if country == "USA":
        stress_periods = [
            ("1929–1932", "Great Depression", "Severe, multi-year equity collapse.", [1929, 1930, 1931, 1932]),
            ("1937", "Recession-era sell-off", "A further sharp market decline during the Depression-era recovery.", [1937]),
            ("1973–1974", "Oil shock and stagflation", "Two consecutive years of negative S&P 500 total returns alongside high inflation.", [1973, 1974]),
            ("2000–2002", "Dot-com bust", "Three consecutive negative calendar-year total returns.", [2000, 2001, 2002]),
            ("2008", "Global Financial Crisis", "Severe equity-market decline during the financial crisis.", [2008]),
            ("2022", "Inflation and rate shock", "Equities fell as inflation and interest rates rose.", [2022]),
        ]
    elif country == "India":
        stress_periods = [
            ("1986–1987", "Early-period market weakness", "Two consecutive negative Sensex total-return financial years.", [1986, 1987]),
            ("1992–1993", "Securities scam / market correction", "A very sharp correction following the 1992 securities-market scandal.", [1992]),
            ("2000–2002", "Dot-com bust and prolonged weakness", "Three consecutive negative Sensex total-return financial years.", [2000, 2001, 2002]),
            ("2008–2009", "Global Financial Crisis", "Severe global risk-off sell-off; the model records the negative FY2008-09 return.", [2008]),
            ("2011–2012", "Slowdown and global risk aversion", "Negative FY2011-12 Sensex total return amid domestic and global pressures.", [2011]),
            ("2015–2016", "Emerging-market and commodity slowdown", "Negative FY2015-16 Sensex total return.", [2015]),
            ("2019–2020", "Pre-COVID slowdown and COVID shock", "The financial year includes the March 2020 crash; the model records a negative FY2019-20 total return.", [2019]),
            ("2025–2026", "FPI selling and geopolitical / oil-price shock", "The embedded FY2025-26 Sensex total return is -6.01%. This is the latest India observation in this file; the Iran-conflict/oil shock is context, not a separately modelled causal factor.", [2025]),
        ]
    else:
        stress_periods = [
            ("1990–1992", "Japanese asset-price bubble collapse", "Three consecutive negative TOPIX total-return years as the late-1980s bubble deflated; 1990 alone is about -39%.", [1990, 1991, 1992]),
            ("1996–1998", "Asian financial crisis and banking stress", "Three consecutive negative annual Japanese equity total returns.", [1996, 1997, 1998]),
            ("2000–2002", "Dot-com bust and prolonged weakness", "Three consecutive negative annual Japanese equity total returns.", [2000, 2001, 2002]),
            ("2007–2008", "Global Financial Crisis", "Two consecutive negative annual Japanese equity total returns.", [2007, 2008]),
            ("2011", "Post-earthquake / global risk aversion", "Negative annual Japanese equity total return.", [2011]),
            ("2018", "Global market sell-off", "Negative annual Japanese equity total return.", [2018]),
        ]

    scenario_windows = [(start, start + years - 1) for start, years in SCENARIOS[country]]
    with st.expander("Historical Stress Periods", expanded=False):
        st.caption("Major negative market periods in the embedded annual total-return data. A period is marked as covered when it overlaps at least one currently tested historical path.")
        for period, title, detail, years in stress_periods:
            returns = []
            for year in years:
                return_data = US if country == "USA" else INDIA if country == "India" else JAPAN
                if year in return_data:
                    annual_return = return_data[year][0] * 100
                    returns.append(f"{ylab(country, year)}: {annual_return:+.2f}%")
            covered = any(any(start <= year <= end for start, end in scenario_windows) for year in years)
            status = "In tested window" if covered else "Outside tested windows"
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{period} · {title}**")
                    st.caption(detail)
                    if returns:
                        # Highlight negative annual returns with a subtle light-red treatment.
                        styled_returns = []
                        for item in returns:
                            match = re.match(r"^(.*?:\s*)([+-]?\d+\.\d+%)$", item)
                            if match and match.group(2).startswith("-"):
                                styled_returns.append(
                                    f'<span style="background-color:#FCE8E8;color:#9B3C3C;'
                                    f'padding:2px 5px;border-radius:4px;font-weight:600;">'
                                    f'{match.group(1)}{match.group(2)}</span>'
                                )
                            else:
                                styled_returns.append(item)
                        st.markdown(
                            "Embedded annual total returns: " + "; ".join(styled_returns),
                            unsafe_allow_html=True,
                        )
                with right:
                    st.caption("🟢 " + status if covered else "⚪ " + status)
        if country == "India":
            st.info("Coverage note: the file contains FY2025-26 (April 2025–March 2026) at -6.01%. FY2024-25 in the same embedded series is +6.39%, so the data does not show two consecutive negative financial years. The model does not separately attribute returns to FPI flows, the Iran conflict, or oil prices.")
        elif country == "USA":
            st.info("Coverage note: US observations are calendar-year S&P 500 total returns. Events outside the selected 1929–1978 and 1966–2015 paths are shown for context but do not affect those simulations.")
        else:
            st.info("Coverage note: Japan equity returns are TOPIX price returns plus dividends. Dividends for 2021–2025 are a 2.3% estimate, and 2026 is excluded because the year is incomplete. The tested paths are 1990–2025 (36 years) and 2000–2025 (26 years).")

    with st.expander("TAX Calculations", expanded=False):
        st.markdown("""
1. **Withdrawal and tax at the beginning of the year:** The planned net withdrawal is calculated, and the selected tax rate is applied to the entire withdrawal.

2. **Both amounts are deducted from the corpus:** The corpus pays the net withdrawal plus the tax.

3. **Remaining corpus stays invested:** The balance earns the year's blended portfolio return.

4. **Equity:** The equity allocation earns the selected country's embedded historical equity return series (S&P 500 for USA, Sensex TRI for India, TOPIX price return plus dividends for Japan). There is no separate tax on equity gains or investment returns in this model.

5. **Debt:** The debt allocation earns the selected debt return:
   - India: Matches Inflation, Inflation −1%, or Inflation −2%.
   - USA: Historical 3-month Treasury bill return.

6. **Withdrawn money earns zero returns:** Both the money withdrawn for expenses and the tax paid are removed at the beginning of the year. Neither participates in that year's investment returns.
        """)

    with st.expander("Data and method"):
        st.markdown(
            "- Withdraw at the start of each year, then the rest earns the blended return. Yearly rebalance. "
            f"Tax is {tax_rate:.0%} of the net withdrawal only; the corpus pays net withdrawal + tax. Tax is not applied to investment returns or the remaining corpus. No fees or other income.\\n"
            + debt_method_text
            + "- Fixed method: the first-year amount rises with each year's inflation.\n"
            "- Taper cuts total spending after age 80 under both withdrawal methods.\n"
            + country_data_method
        )



# ======================================================================
# APP SWITCH
# ======================================================================
VIEWS = ["Monte Carlo", "Historical Stress Test"]
# Widget keys that must survive while their tab is hidden (Streamlit drops the state of widgets that are not drawn).
STRESS_KEYS = ["country", "method", "st_corpus_cr", "st_spend_cr", "corpus_w_India", "spend_w_India", "corpus_w_Japan",
               "spend_w_Japan", "corpus_w_USA", "spend_w_USA", "st_taper", "st_equity", "st_debt", "st_tax"]

# ----------------------------------------------------------------------
# Shared inputs: corpus, total spend and equity % follow you from one tab to the other.
# Canonical units: corpus and total spend in rupee crore, equity in %.
# ----------------------------------------------------------------------
DEBT_CHOICES = {"Matches Inflation": 0.0, "-1%": -1.0, "-2%": -2.0}
SYNC_FIELDS = ["corpus", "spend", "equity"]   # returns and debt are NOT shared: history uses actuals, Monte Carlo uses its own sliders


def _read_side(side: str) -> dict:
    ss = st.session_state
    if side == "MC":
        return {"corpus": float(ss["corpus_cr"]),
                "spend": (float(ss["essential_l"]) + float(ss["discretionary_l"])) / 100.0,  # medical is not included
                "equity": float(ss["equity_pct"]),
                "debt": float(ss["debt_return_pct"]) - float(ss["inflation_pct"])}
    return {"corpus": float(ss["st_corpus_cr"]), "spend": float(ss["st_spend_cr"]),
            "equity": float(ss["st_equity"]), "debt": DEBT_CHOICES[ss["st_debt"]]}


def _write_side(side: str, field: str, v: float) -> None:
    ss = st.session_state
    if side == "MC":
        if field == "corpus":
            ss["corpus_cr"] = round(min(100.0, max(v, 0.05)), 2)
        elif field == "spend":
            ess, disc = float(ss["essential_l"]), float(ss["discretionary_l"])
            total_l = max(v * 100.0, 0.0)
            share = ess / (ess + disc) if (ess + disc) > 0 else 1.0
            ss["essential_l"] = min(150.0, round(total_l * share, 2))
            ss["discretionary_l"] = min(150.0, round(total_l - round(total_l * share, 2), 2))
        elif field == "equity":
            ss["equity_pct"] = int(min(100, max(0, round(v / 5.0) * 5)))
        elif field == "debt":
            ss["debt_return_pct"] = float(min(20.0, max(-10.0, round((float(ss["inflation_pct"]) + v) * 4) / 4)))
    else:
        if field == "corpus":
            ss["st_corpus_cr"] = v
        elif field == "spend":
            ss["st_spend_cr"] = v
        elif field == "equity":
            ss["st_equity"] = int(min(100, max(0, round(v / 5.0) * 5)))
        elif field == "debt":
            for name, d in DEBT_CHOICES.items():   # only snap when the value is one of the three choices
                if abs(d - v) <= 0.5:
                    ss["st_debt"] = name


def sync_tabs(view: str) -> None:
    """On the first run, start the Historical Stress Test from the Monte Carlo values. After that, whenever you switch
    tabs, carry across only the fields you changed in the tab you are leaving."""
    ss = st.session_state
    init_state()
    ss.setdefault("st_corpus_cr", 5.0)
    ss.setdefault("st_spend_cr", 0.24)
    ss.setdefault("st_equity", 60)
    ss.setdefault("st_debt", "Matches Inflation")
    prev = ss.get("_active_view")
    if prev is None:
        for fld in SYNC_FIELDS:
            _write_side("ST", fld, _read_side("MC")[fld])
        ss["_seen"] = {"MC": _read_side("MC"), "ST": _read_side("ST")}
    elif prev != view:
        src, dst = ("MC", "ST") if prev == "Monte Carlo" else ("ST", "MC")
        seen, cur = ss["_seen"], _read_side(src)
        for fld in SYNC_FIELDS:
            if abs(cur[fld] - seen[src][fld]) > 1e-9:       # changed here since the last hand-over
                _write_side(dst, fld, cur[fld])
        ss["_seen"] = {src: _read_side(src), dst: _read_side(dst)}
    ss["_active_view"] = view


def keep_state_alive() -> None:
    """Re-save every stored input each run so a hidden tab keeps its values when you come back."""
    keys = list(DEFAULTS) + ["nav_section", "explain_topic", "corpus_cr_w", "essential_l_w", "discretionary_l_w", "_last_view", "_active_view", "_seen"] + STRESS_KEYS
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def app() -> None:
    # Lock the light theme: both tools use light colours written into the code, so Streamlit's automatic dark theme
    # makes them unreadable. The theme is chosen when a browser session starts, so the very first session after
    # the server starts reloads once (about a second); every later session starts light.
    try:
        if st._config.get_option("theme.base") != "light":
            st._config.set_option("theme.base", "light")
            import streamlit.components.v1 as components
            components.html("<script>window.parent.location.reload();</script>", height=0)
            st.stop()
    except Exception:
        pass
    st.set_page_config(page_title="Retirement Corpus StressTest", layout="wide")
    keep_state_alive()
    with st.sidebar:
        if hasattr(st, "segmented_control"):
            # Tab-style switch. A click on the active tab would clear it, so fall back to the last choice.
            st.markdown("""<style>
.st-key-app_view [data-testid="stButtonGroup"], .st-key-app_view [role="group"] { flex-wrap: nowrap !important; width: 100%; gap: 0 !important; }
.st-key-app_view button { flex: 1 1 auto !important; white-space: nowrap !important; padding: .3rem .3rem !important; min-width: 0 !important; }
.st-key-app_view button[data-testid*="Active"], .st-key-app_view button[aria-checked="true"], .st-key-app_view button[aria-pressed="true"] { background-color: #DCFCE7 !important; border-color: #16A34A !important; }
.st-key-app_view button p, .st-key-app_view button span { font-size: .88rem !important; color: #111827 !important; font-weight: 400 !important; white-space: nowrap !important; }
</style>""", unsafe_allow_html=True)
            st.session_state.setdefault("app_view", VIEWS[0])
            picked = st.segmented_control("View", VIEWS, key="app_view", label_visibility="collapsed")
            view = picked or st.session_state.get("_last_view", VIEWS[0])
        else:
            view = st.radio("View", VIEWS, horizontal=True, key="app_view", label_visibility="collapsed")
        st.session_state["_last_view"] = view
        st.divider()
    sync_tabs(view)
    if view == "Historical Stress Test":
        run_stress_test()
    else:
        main()


if __name__ == "__main__":
    # Streamlit apps should be launched with `streamlit run`, not plain `python`.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None
    if _ctx is None:
        print(
            "This is a Streamlit app. Launch it with:\n"
            '  streamlit run "Retirement Corpus StressTest.py"\n'
            "If Streamlit is missing, install it with: python -m pip install streamlit"
        )
        raise SystemExit(0)
    app()
