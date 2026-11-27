"""
Retirement Stress Test (Streamlit)
Replays historical USA, India and Japan return sequences to test retirement sustainability.

Run:   pip install streamlit pandas altair
       streamlit run retirement_Stresstest_app_FY_returns_dual_returns.py

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
"""
import altair as alt
import pandas as pd
import streamlit as st

import html
import re
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


if True:
    st.set_page_config(page_title="Retirement Stress Test", layout="wide")
    # Lock the light theme: the page colours are written for a light background, so Streamlit's automatic dark theme makes it
    # unreadable. The theme is chosen when a browser session starts, so the first session after the server starts
    # reloads once (about a second); every later session starts light.
    try:
        if st._config.get_option("theme.base") != "light":
            st._config.set_option("theme.base", "light")
            import streamlit.components.v1 as components
            components.html("<script>window.parent.location.reload();</script>", height=0)
            st.stop()
    except Exception:
        pass
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
    st.title("Retirement Stress Test")
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
        ss.setdefault("corpus_cr", 5.0)
        ss.setdefault("spend_cr", 0.24)
        if country == "USA":
            usd_cr = lambda m: m * 1e6 * USD_INR / 1e7          # $ million -> crore
            usd_sp = lambda k: k * 1e3 * USD_INR / 1e7          # $ thousand -> crore
            ss["corpus_w_USA"] = nearest(USD_CORPUS_OPTIONS, usd_cr, ss["corpus_cr"])
            ss["spend_w_USA"] = nearest(USD_SPEND_OPTIONS, usd_sp, ss["spend_cr"])
            corpus_usd = st.select_slider("Starting corpus", options=USD_CORPUS_OPTIONS,
                                          format_func=lambda m: f"${m:.1f} M ({money(usd_cr(m))})", key="corpus_w_USA",
                                          on_change=lambda: ss.update(corpus_cr=usd_cr(ss["corpus_w_USA"])))
            spend_usd = st.select_slider("First-year expenses (per year)", options=USD_SPEND_OPTIONS,
                                         format_func=lambda k: f"${k} K ({money(usd_sp(k))})", key="spend_w_USA",
                                         on_change=lambda: ss.update(spend_cr=usd_sp(ss["spend_w_USA"])))
            corpus, spend_lakh = usd_cr(corpus_usd), usd_sp(spend_usd) * 100
        else:
            country_key = "India" if country == "India" else "Japan"
            ss[f"corpus_w_{country_key}"] = nearest(CORPUS_OPTIONS, lambda o: o, ss["corpus_cr"])
            ss[f"spend_w_{country_key}"] = nearest(SPEND_OPTIONS, lambda o: o / 100, ss["spend_cr"])
            corpus = st.select_slider("Starting corpus", options=CORPUS_OPTIONS,
                                      format_func=lambda v: f"₹{v:g} Cr", key=f"corpus_w_{country_key}",
                                      on_change=lambda: ss.update(corpus_cr=ss[f"corpus_w_{country_key}"]))
            spend_lakh = st.select_slider("First-year expenses (per year)", options=SPEND_OPTIONS,
                                          format_func=lambda v: f"₹{v} L" if v < 100 else f"₹{v / 100:g} Cr", key=f"spend_w_{country_key}",
                                          on_change=lambda: ss.update(spend_cr=ss[f"spend_w_{country_key}"] / 100))
        spend = spend_lakh / 100  # in crore
        taper = st.slider("Taper after age 80 (% per year)", 0.0, 5.0, 2.0, 0.5) / 100
        age0 = START_AGE
        equity = st.slider("Equity share (%)", 0, 100, 60, 5) / 100
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
