"""Auto-generated business insights from the (cleaned) dataset."""
import pandas as pd
from utils.data_loader import coerce_likert, ORDERED, label, explode_multiselect, LIKERT_COLS


def _top(df, col):
    s = df[col].dropna().astype(str)
    if s.empty:
        return "—", 0
    vc = s.value_counts(normalize=True)
    return vc.index[0], round(100 * vc.iloc[0], 1)


def _share(df, col, values):
    s = df[col].dropna().astype(str)
    if s.empty:
        return 0
    return round(100 * s.isin(values).mean(), 1)


def generate(df):
    d = coerce_likert(df)
    n = len(df)
    out = {}

    # Demographics
    age, age_p = _top(df, "age_group")
    emirate, em_p = _top(df, "emirate")
    nat, nat_p = _top(df, "nationality")
    inc, inc_p = _top(df, "income")
    out["Demographics"] = [
        f"The sample of **{n:,}** respondents skews toward the **{age}** age band ({age_p}%).",
        f"**{emirate}** is the most represented emirate ({em_p}%), and **{nat}** the largest nationality group ({nat_p}%).",
        f"The most common monthly income band is **{inc}** ({inc_p}%).",
    ]

    # Shopping behaviour
    chan, chan_p = _top(df, "shopping_channel")
    loc, loc_p = _top(df, "buy_location")
    freq, freq_p = _top(df, "purchase_freq")
    out["Shopping Behaviour"] = [
        f"Preferred shopping channel: **{chan}** ({chan_p}%); most common purchase point: **{loc}** ({loc_p}%).",
        f"Most respondents buy accessories **{freq.lower()}** ({freq_p}%).",
    ]

    # Spending
    tie_premium = _share(df, "spend_tie", ["AED 201–300", "Above AED 300"])
    socks_premium = _share(df, "spend_socks", ["AED 51–75", "Above AED 75"])
    out["Spending Behaviour"] = [
        f"**{tie_premium}%** would spend AED 200+ on a premium tie.",
        f"**{socks_premium}%** would spend AED 50+ on premium dress socks.",
    ]

    # Purchase intent
    intent = _share(df, "purchase_likelihood_6m", ["Likely", "Very Likely"])
    consider = _share(df, "saturn_consideration", ["Probably Would", "Definitely Would"])
    willing = _share(df, "willing_try_new", ["Definitely", "Probably"])
    out["Purchase Intent"] = [
        f"**{intent}%** are likely or very likely to buy premium accessories in the next 6 months.",
        f"**{consider}%** would probably or definitely consider Saturn.",
        f"**{willing}%** are open to trying a new premium UAE brand.",
    ]

    # Importance drivers (mean Likert)
    means = {label(c): round(d[c].mean(), 2) for c in LIKERT_COLS if c in d.columns}
    means = dict(sorted(means.items(), key=lambda x: x[1], reverse=True))
    top3 = list(means.items())[:3]
    out["Key Purchase Drivers"] = [
        f"Highest-rated drivers (mean importance 1–5): "
        + ", ".join(f"**{k}** ({v})" for k, v in top3) + "."
    ]

    # What convinces trial
    conv = explode_multiselect(df, "convince_try").value_counts(normalize=True).head(3)
    out["Trial Triggers"] = [
        "Top motivators to try a new brand: "
        + ", ".join(f"**{i}** ({round(100*p,1)}%)" for i, p in conv.items()) + "."
    ]

    # Recommendations
    drivers = ", ".join(k.replace("Importance: ", "") for k, _ in top3)
    out["Recommendations"] = [
        f"Lead product messaging with **{drivers}** — these are the attributes customers rate most highly.",
        f"With **{tie_premium}%** willing to pay AED 200+ for ties, anchor a clear premium tier while keeping an accessible entry price.",
        f"Prioritise the **{chan}** channel for launch, supported by the strongest trial triggers above.",
        f"Convert the **{willing}%** open-to-new segment with introductory offers, premium-quality proof and visible social/word-of-mouth reviews.",
        "Use the K-Means segments (Predictive Analytics) to tailor pricing tiers and creative to high-propensity vs price-sensitive groups.",
    ]

    # Risks
    price_sensitive = _share(df, "main_influence", ["Price"])
    out["Business Risks"] = [
        f"**{price_sensitive}%** cite price as their single biggest influence — a premium-only stance risks alienating them.",
        "Established names (Hugo Boss, Massimo Dutti, Zara) already hold mindshare; differentiation must be explicit.",
    ]
    return out


def executive_summary(df, quality=None):
    d = coerce_likert(df)
    n = len(df)
    consider = _share(df, "saturn_consideration", ["Probably Would", "Definitely Would"])
    tie_premium = _share(df, "spend_tie", ["AED 201–300", "Above AED 300"])
    chan, _ = _top(df, "shopping_channel")
    q = f" The cleaned dataset scored **{quality}%** on data quality." if quality else ""
    return (
        f"Across **{n:,}** UAE consumers, **{consider}%** would consider purchasing from Saturn, "
        f"and **{tie_premium}%** are willing to pay a premium (AED 200+) for ties. "
        f"Demand concentrates in higher-income, formally-dressed professionals who shop primarily via **{chan}**. "
        f"The opportunity is a clearly-tiered premium accessories brand that leads on fabric quality, design and brand trust "
        f"while retaining an accessible entry price to capture the sizeable value-conscious segment.{q}"
    )
