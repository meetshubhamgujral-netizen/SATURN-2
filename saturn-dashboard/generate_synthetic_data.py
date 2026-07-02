"""
Saturn Consumer Understanding Survey (UAE)
Synthetic data generator
==========================

Produces a realistic, *correlated* survey dataset of 3,000 respondents that maps
1:1 to the 30-question Saturn questionnaire, then deliberately injects ~10% of
"bad" / dirty responses so the dataset requires real data cleaning.

The clean signal is driven by a hidden latent "premium propensity" score built
from income, occupation, formality and age. That score tilts spend, brand
choice, importance ratings and purchase intent so that the analytics, correlation
and clustering pages later have genuine structure to discover.

Run:
    python generate_synthetic_data.py
Output:
    data/saturn_survey_raw.csv      (the dirty dataset to be cleaned)
    data/_data_dictionary.csv       (column reference)

Injected data-quality issues (for the Data Cleaning page to catch):
    1. Missing values            ~120 rows
    2. Likert straight-lining    ~60  rows  (all importance items identical)
    3. Out-of-range Likert       ~35  rows  (0, 6, 7, -1, "N/A")
    4. Logical contradictions    ~50  rows  (e.g. "Never" + buys monthly)
    5. Whitespace / casing noise ~55  rows  (" dubai ", "MALE", etc.)
    6. Speeder responses         ~45  rows  (duration < 25s)
    7. Mixed-type numeric        ~25  rows  ("five", "3 " in Likert columns)
    8. Junk open-ended text      ~45  rows  ("asdasd", "....", "123")
    9. Exact duplicate rows      ~40  rows  (appended -> breaks unique id)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000

# --------------------------------------------------------------------------- #
# Helper: weighted categorical draw
# --------------------------------------------------------------------------- #
def pick(options, probs, size=N):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return RNG.choice(options, size=size, p=probs)


def tier_pick(tier, table):
    """Draw one value per row using a {tier: (options, probs)} table."""
    out = np.empty(len(tier), dtype=object)
    for t, (opts, p) in table.items():
        mask = tier == t
        out[mask] = pick(opts, p, size=mask.sum())
    return out


# --------------------------------------------------------------------------- #
# Section A — Demographics (base distributions tuned to the UAE)
# --------------------------------------------------------------------------- #
age_group = pick(
    ["Under 18", "18–24", "25–34", "35–44", "45–54", "55+"],
    [0.02, 0.14, 0.38, 0.28, 0.13, 0.05],
)
gender = pick(["Male", "Female", "Prefer not to say"], [0.80, 0.17, 0.03])
emirate = pick(
    ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"],
    [0.42, 0.28, 0.16, 0.05, 0.04, 0.03, 0.02],
)
nationality = pick(
    ["South Asian", "Arab Expat", "Emirati", "Southeast Asian", "European", "GCC", "African", "Other"],
    [0.38, 0.16, 0.12, 0.10, 0.09, 0.06, 0.05, 0.04],
)
occupation = pick(
    ["Private Employee", "Government Employee", "Business Owner", "Self-employed", "Student", "Other"],
    [0.44, 0.14, 0.12, 0.12, 0.11, 0.07],
)

# Income depends on occupation + age (built via additive logits then bucketed)
occ_income = {
    "Private Employee": 0.4, "Government Employee": 0.8, "Business Owner": 1.4,
    "Self-employed": 0.6, "Student": -1.4, "Other": -0.2,
}
age_income = {
    "Under 18": -1.5, "18–24": -0.6, "25–34": 0.3, "35–44": 0.8, "45–54": 0.7, "55+": 0.2,
}
inc_logit = (
    np.array([occ_income[o] for o in occupation])
    + np.array([age_income[a] for a in age_group])
    + RNG.normal(0, 0.9, N)
)
# Map logit to 5 income bands by quantile thresholds
q = np.quantile(inc_logit, [0.12, 0.38, 0.68, 0.88])
income = np.select(
    [inc_logit < q[0], inc_logit < q[1], inc_logit < q[2], inc_logit < q[3]],
    ["Under 5,000", "5,000–10,000", "10,001–20,000", "20,001–35,000"],
    default="Above 35,000",
)

# --------------------------------------------------------------------------- #
# Section B — Fashion & Lifestyle
# --------------------------------------------------------------------------- #
occ_formal = {
    "Private Employee": 0.7, "Government Employee": 1.1, "Business Owner": 1.3,
    "Self-employed": 0.4, "Student": -0.6, "Other": 0.0,
}
formal_logit = np.array([occ_formal[o] for o in occupation]) + RNG.normal(0, 0.8, N)
qf = np.quantile(formal_logit, [0.18, 0.45, 0.68, 0.86])
formal_attire_freq = np.select(
    [formal_logit < qf[0], formal_logit < qf[1], formal_logit < qf[2], formal_logit < qf[3]],
    ["Rarely", "Occasionally", "Once a week", "3–5 times/week"],
    default="Daily",
)

# --------------------------------------------------------------------------- #
# Latent "premium propensity" score -> drives the rest of the survey
# --------------------------------------------------------------------------- #
inc_rank = {"Under 5,000": 0, "5,000–10,000": 1, "10,001–20,000": 2, "20,001–35,000": 3, "Above 35,000": 4}
formal_rank = {"Rarely": 0, "Occasionally": 1, "Once a week": 2, "3–5 times/week": 3, "Daily": 4}
premium = (
    1.15 * np.array([inc_rank[i] for i in income])
    + 0.85 * np.array([formal_rank[f] for f in formal_attire_freq])
    + np.array([occ_income[o] for o in occupation])
    + RNG.normal(0, 1.0, N)
)
tier = pd.qcut(premium, 3, labels=["Low", "Mid", "High"]).astype(object)

# Q10 purchase frequency
purchase_freq = tier_pick(tier, {
    "Low":  (["Monthly", "Every 3 months", "Every 6 months", "Once a year", "Only when needed"], [0.04, 0.12, 0.18, 0.26, 0.40]),
    "Mid":  (["Monthly", "Every 3 months", "Every 6 months", "Once a year", "Only when needed"], [0.10, 0.24, 0.26, 0.18, 0.22]),
    "High": (["Monthly", "Every 3 months", "Every 6 months", "Once a year", "Only when needed"], [0.22, 0.34, 0.22, 0.10, 0.12]),
})

# --------------------------------------------------------------------------- #
# Multi-select helpers
# --------------------------------------------------------------------------- #
def multiselect(prob_map, tier_arr, none_label=None, none_prob_map=None, max_pick=None):
    """prob_map: {option: {tier: p}}. Returns list of '; '-joined strings."""
    rows = []
    for idx, t in enumerate(tier_arr):
        chosen = [opt for opt, pm in prob_map.items() if RNG.random() < pm[t]]
        if max_pick and len(chosen) > max_pick:
            chosen = list(RNG.choice(chosen, size=max_pick, replace=False))
        if none_label is not None and none_prob_map is not None:
            if RNG.random() < none_prob_map[t]:
                chosen = [none_label]
        if not chosen:
            chosen = [none_label] if none_label else [list(prob_map.keys())[0]]
        # keep original option order for tidiness
        order = list(prob_map.keys()) + ([none_label] if none_label else [])
        chosen = [o for o in order if o in chosen]
        rows.append("; ".join(chosen))
    return rows

# Q8 occasions
occasions = multiselect(
    {
        "Office":            {"Low": 0.45, "Mid": 0.62, "High": 0.72},
        "Weddings":          {"Low": 0.40, "Mid": 0.55, "High": 0.66},
        "Business Meetings": {"Low": 0.20, "Mid": 0.45, "High": 0.70},
        "Parties":           {"Low": 0.30, "Mid": 0.38, "High": 0.46},
        "Religious Events":  {"Low": 0.35, "Mid": 0.40, "High": 0.42},
        "Formal Dinners":    {"Low": 0.18, "Mid": 0.35, "High": 0.58},
    },
    tier, none_label="Never",
    none_prob_map={"Low": 0.10, "Mid": 0.03, "High": 0.01},
)

# Q9 accessories purchased
accessories = multiselect(
    {
        "Ties":          {"Low": 0.40, "Mid": 0.62, "High": 0.78},
        "Pocket Squares":{"Low": 0.14, "Mid": 0.32, "High": 0.55},
        "Dress Socks":   {"Low": 0.48, "Mid": 0.60, "High": 0.70},
        "Handkerchiefs": {"Low": 0.22, "Mid": 0.30, "High": 0.40},
        "Cufflinks":     {"Low": 0.12, "Mid": 0.28, "High": 0.52},
        "Belts":         {"Low": 0.55, "Mid": 0.62, "High": 0.66},
    },
    tier, none_label="None",
    none_prob_map={"Low": 0.08, "Mid": 0.02, "High": 0.01},
)

# --------------------------------------------------------------------------- #
# Section C — Buying Behaviour
# --------------------------------------------------------------------------- #
buy_location = tier_pick(tier, {
    "Low":  (["Brand Store", "Shopping Mall", "Department Store", "Online Marketplace", "Brand Website", "Tailor/Boutique"], [0.08, 0.30, 0.18, 0.32, 0.08, 0.04]),
    "Mid":  (["Brand Store", "Shopping Mall", "Department Store", "Online Marketplace", "Brand Website", "Tailor/Boutique"], [0.16, 0.26, 0.18, 0.22, 0.12, 0.06]),
    "High": (["Brand Store", "Shopping Mall", "Department Store", "Online Marketplace", "Brand Website", "Tailor/Boutique"], [0.28, 0.18, 0.14, 0.12, 0.16, 0.12]),
})

# Younger -> more online
young = np.isin(age_group, ["Under 18", "18–24", "25–34"])
shopping_channel = np.where(
    young,
    pick(["Physical Store", "Online", "No Preference"], [0.34, 0.50, 0.16], size=N),
    pick(["Physical Store", "Online", "No Preference"], [0.52, 0.30, 0.18], size=N),
)

spend_tie = tier_pick(tier, {
    "Low":  (["Under AED 50", "AED 50–100", "AED 101–200", "AED 201–300", "Above AED 300"], [0.40, 0.36, 0.18, 0.04, 0.02]),
    "Mid":  (["Under AED 50", "AED 50–100", "AED 101–200", "AED 201–300", "Above AED 300"], [0.16, 0.34, 0.32, 0.12, 0.06]),
    "High": (["Under AED 50", "AED 50–100", "AED 101–200", "AED 201–300", "Above AED 300"], [0.05, 0.16, 0.34, 0.27, 0.18]),
})

spend_socks = tier_pick(tier, {
    "Low":  (["Under AED 25", "AED 25–50", "AED 51–75", "Above AED 75"], [0.52, 0.34, 0.10, 0.04]),
    "Mid":  (["Under AED 25", "AED 25–50", "AED 51–75", "Above AED 75"], [0.30, 0.42, 0.20, 0.08]),
    "High": (["Under AED 25", "AED 25–50", "AED 51–75", "Above AED 75"], [0.14, 0.38, 0.30, 0.18]),
})

main_influence = tier_pick(tier, {
    "Low":  (["Price", "Fabric Quality", "Brand Reputation", "Design", "Comfort", "Durability"], [0.40, 0.12, 0.06, 0.14, 0.18, 0.10]),
    "Mid":  (["Price", "Fabric Quality", "Brand Reputation", "Design", "Comfort", "Durability"], [0.22, 0.22, 0.14, 0.18, 0.14, 0.10]),
    "High": (["Price", "Fabric Quality", "Brand Reputation", "Design", "Comfort", "Durability"], [0.08, 0.30, 0.24, 0.20, 0.10, 0.08]),
})

# --------------------------------------------------------------------------- #
# Section D — Importance Likert (1-5), correlated with premium score
# --------------------------------------------------------------------------- #
def likert(center_low, center_high, spread=1.0, invert_tier=False):
    """Center shifts by tier; clipped to 1..5 integers."""
    centers = {"Low": center_low, "Mid": (center_low + center_high) / 2, "High": center_high}
    if invert_tier:
        centers = {"Low": center_high, "Mid": (center_low + center_high) / 2, "High": center_low}
    vals = np.array([RNG.normal(centers[t], spread) for t in tier])
    return np.clip(np.round(vals), 1, 5).astype(int)

imp_fabric_quality = likert(3.2, 4.6)
imp_design         = likert(3.3, 4.4)
imp_pricing        = likert(4.4, 3.0, invert_tier=True)   # price matters more to Low tier
imp_brand_rep      = likert(2.8, 4.5)
imp_comfort        = likert(3.8, 4.2, spread=0.9)
imp_durability     = likert(3.7, 4.3, spread=0.9)
imp_variety        = likert(3.2, 3.9)
imp_sustainability = likert(2.6, 3.6, spread=1.2)

# --------------------------------------------------------------------------- #
# Section E — Brand & Market Perception
# --------------------------------------------------------------------------- #
brands_purchased = multiselect(
    {
        "Zara":             {"Low": 0.46, "Mid": 0.40, "High": 0.24},
        "H&M":              {"Low": 0.44, "Mid": 0.30, "High": 0.14},
        "Massimo Dutti":    {"Low": 0.16, "Mid": 0.34, "High": 0.46},
        "Hugo Boss":        {"Low": 0.08, "Mid": 0.26, "High": 0.52},
        "Charles Tyrwhitt": {"Low": 0.05, "Mid": 0.16, "High": 0.34},
        "Brooks Brothers":  {"Low": 0.04, "Mid": 0.14, "High": 0.32},
    },
    tier, none_label="Other",
    none_prob_map={"Low": 0.12, "Mid": 0.08, "High": 0.05},
)

loyalty_factor = tier_pick(tier, {
    "Low":  (["Quality", "Price", "Design", "Customer Service", "Prestige", "Product Variety"], [0.22, 0.36, 0.14, 0.12, 0.04, 0.12]),
    "Mid":  (["Quality", "Price", "Design", "Customer Service", "Prestige", "Product Variety"], [0.34, 0.18, 0.16, 0.14, 0.08, 0.10]),
    "High": (["Quality", "Price", "Design", "Customer Service", "Prestige", "Product Variety"], [0.40, 0.08, 0.16, 0.14, 0.16, 0.06]),
})

willing_try_new = tier_pick(tier, {
    "Low":  (["Definitely", "Probably", "Not Sure", "Probably Not", "Definitely Not"], [0.14, 0.30, 0.30, 0.18, 0.08]),
    "Mid":  (["Definitely", "Probably", "Not Sure", "Probably Not", "Definitely Not"], [0.22, 0.40, 0.24, 0.10, 0.04]),
    "High": (["Definitely", "Probably", "Not Sure", "Probably Not", "Definitely Not"], [0.34, 0.40, 0.18, 0.06, 0.02]),
})

convince_try = multiselect(
    {
        "Introductory Discounts":     {"Low": 0.62, "Mid": 0.50, "High": 0.34},
        "Premium Quality":            {"Low": 0.34, "Mid": 0.52, "High": 0.70},
        "Social Media Reviews":       {"Low": 0.40, "Mid": 0.42, "High": 0.38},
        "Influencer Recommendations": {"Low": 0.30, "Mid": 0.30, "High": 0.26},
        "Word of Mouth":              {"Low": 0.36, "Mid": 0.42, "High": 0.46},
        "Luxury Packaging":           {"Low": 0.14, "Mid": 0.24, "High": 0.42},
        "Money-back Guarantee":       {"Low": 0.38, "Mid": 0.40, "High": 0.40},
    },
    tier, max_pick=3,
)

# --------------------------------------------------------------------------- #
# Section F — Purchase Intention (depends on premium + openness to new brands)
# --------------------------------------------------------------------------- #
try_rank = {"Definitely": 2, "Probably": 1, "Not Sure": 0, "Probably Not": -1, "Definitely Not": -2}
intent = (
    0.8 * (premium - premium.mean()) / premium.std()
    + 0.9 * np.array([try_rank[w] for w in willing_try_new])
    + RNG.normal(0, 0.8, N)
)
qi = np.quantile(intent, [0.16, 0.42, 0.68, 0.88])
purchase_likelihood_6m = np.select(
    [intent < qi[0], intent < qi[1], intent < qi[2], intent < qi[3]],
    ["Very Unlikely", "Unlikely", "Neutral", "Likely"], default="Very Likely",
)
qs = np.quantile(intent, [0.14, 0.40, 0.66, 0.87])
saturn_consideration = np.select(
    [intent < qs[0], intent < qs[1], intent < qs[2], intent < qs[3]],
    ["Definitely Would Not", "Probably Would Not", "Not Sure", "Probably Would"],
    default="Definitely Would",
)

# Q30 open-ended
feature_pool = [
    "Premium fabric quality", "Affordable pricing", "Unique and elegant designs",
    "Strong brand reputation", "Durability and long-lasting material",
    "Good customer service", "Fast and reliable delivery", "Luxury packaging",
    "Wide variety of designs", "Comfort and fit", "Local UAE identity",
    "Sustainable materials", "Value for money", "Exclusive limited editions",
    "Trustworthy after-sales support",
]
most_important_feature = pick(feature_pool, [1] * len(feature_pool))

# --------------------------------------------------------------------------- #
# Metadata: timing for speeder detection
# --------------------------------------------------------------------------- #
duration_seconds = np.round(RNG.normal(360, 140, N)).astype(int).clip(60, 1200)
start = pd.Timestamp("2025-02-01")
submission_time = [
    (start + pd.Timedelta(minutes=int(RNG.integers(0, 60 * 24 * 60)))).strftime("%Y-%m-%d %H:%M")
    for _ in range(N)
]

df = pd.DataFrame({
    "respondent_id": [f"R{1000 + i}" for i in range(N)],
    "submission_time": submission_time,
    "duration_seconds": duration_seconds,
    "age_group": age_group,
    "gender": gender,
    "emirate": emirate,
    "nationality": nationality,
    "income": income,
    "occupation": occupation,
    "formal_attire_freq": formal_attire_freq,
    "occasions": occasions,
    "accessories_purchased": accessories,
    "purchase_freq": purchase_freq,
    "buy_location": buy_location,
    "shopping_channel": shopping_channel,
    "spend_tie": spend_tie,
    "spend_socks": spend_socks,
    "main_influence": main_influence,
    "imp_fabric_quality": imp_fabric_quality,
    "imp_design": imp_design,
    "imp_pricing": imp_pricing,
    "imp_brand_rep": imp_brand_rep,
    "imp_comfort": imp_comfort,
    "imp_durability": imp_durability,
    "imp_variety": imp_variety,
    "imp_sustainability": imp_sustainability,
    "brands_purchased": brands_purchased,
    "loyalty_factor": loyalty_factor,
    "willing_try_new": willing_try_new,
    "convince_try": convince_try,
    "purchase_likelihood_6m": purchase_likelihood_6m,
    "saturn_consideration": saturn_consideration,
    "most_important_feature": most_important_feature,
})

LIKERT_COLS = [
    "imp_fabric_quality", "imp_design", "imp_pricing", "imp_brand_rep",
    "imp_comfort", "imp_durability", "imp_variety", "imp_sustainability",
]

# ========================================================================== #
#                       NOISE / DIRTY-DATA INJECTION (~10%)                   #
# ========================================================================== #
df = df.astype(object)  # allow mixed types / strings in numeric cols
dirty = set()

def sample_idx(k):
    return RNG.choice(N, size=k, replace=False)

# 1) Missing values (1-3 random non-id fields per row)
nullable = [c for c in df.columns if c not in ("respondent_id",)]
for i in sample_idx(80):
    for c in RNG.choice(nullable, size=int(RNG.integers(1, 4)), replace=False):
        df.at[i, c] = np.nan
    dirty.add(i)

# 2) Likert straight-lining (all 8 items identical)
for i in sample_idx(28):
    v = int(RNG.choice([1, 3, 5]))
    for c in LIKERT_COLS:
        df.at[i, c] = v
    dirty.add(i)

# 3) Out-of-range Likert values
for i in sample_idx(22):
    for c in RNG.choice(LIKERT_COLS, size=int(RNG.integers(1, 3)), replace=False):
        df.at[i, c] = RNG.choice([0, 6, 7, -1, "N/A"])
    dirty.add(i)

# 4) Logical contradictions
for i in sample_idx(32):
    kind = RNG.integers(0, 3)
    if kind == 0:                                   # "Never" yet buys monthly
        df.at[i, "occasions"] = "Never"
        df.at[i, "purchase_freq"] = "Monthly"
        df.at[i, "accessories_purchased"] = "Ties; Belts"
    elif kind == 1:                                 # buys nothing yet spends big
        df.at[i, "accessories_purchased"] = "None"
        df.at[i, "spend_tie"] = "Above AED 300"
    else:                                           # rejects new brands yet would buy Saturn
        df.at[i, "willing_try_new"] = "Definitely Not"
        df.at[i, "saturn_consideration"] = "Definitely Would"
    dirty.add(i)

# 5) Whitespace / casing noise on categoricals
case_cols = ["emirate", "gender", "nationality", "shopping_channel"]
def mangle(s):
    if not isinstance(s, str):
        return s
    r = RNG.random()
    if r < 0.34:   return s.upper()
    if r < 0.67:   return s.lower()
    return f"  {s} "
for i in sample_idx(36):
    c = RNG.choice(case_cols)
    df.at[i, c] = mangle(df.at[i, c])
    dirty.add(i)

# 6) Speeders (implausibly fast completion)
for i in sample_idx(28):
    df.at[i, "duration_seconds"] = int(RNG.integers(5, 25))
    dirty.add(i)

# 7) Mixed-type values inside numeric Likert columns
words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
for i in sample_idx(16):
    c = RNG.choice(LIKERT_COLS)
    cur = df.at[i, c]
    df.at[i, c] = words.get(cur if isinstance(cur, int) else 3, "three") if RNG.random() < 0.5 else "3 "
    dirty.add(i)

# 8) Junk open-ended text
junk = ["asdasd", "....", "123", "n/a", "N/A", "-", "nothing", "...", "qwerty", "."]
for i in sample_idx(28):
    df.at[i, "most_important_feature"] = RNG.choice(junk)
    dirty.add(i)

# 9) Exact duplicate rows (appended -> also breaks the unique respondent_id)
dupe_src = sample_idx(30)
dupes = df.loc[dupe_src].copy()
df = pd.concat([df, dupes], ignore_index=True)

# Shuffle so dirty rows aren't clustered, then save
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

OUT = "data/saturn_survey_raw.csv"
df.to_csv(OUT, index=False)

# Data dictionary
dd = pd.DataFrame({
    "column": df.columns,
    "type": ["metadata"] * 3 + ["categorical"] * 15 + ["likert_1_5"] * 8
            + ["multiselect", "categorical", "categorical", "multiselect",
               "categorical", "categorical", "open_ended"],
})
dd.to_csv("data/_data_dictionary.csv", index=False)

print(f"Rows written : {len(df)} (3000 base + {len(dupe_src)} duplicates)")
print(f"Columns      : {df.shape[1]}")
print(f"Dirty rows   : ~{len(dirty)} corrupted + {len(dupe_src)} duplicates "
      f"(~{100*(len(dirty)+len(dupe_src))/len(df):.1f}% of file)")
print(f"Saved        : {OUT}")
