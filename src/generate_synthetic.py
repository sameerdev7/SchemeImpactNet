"""
generate_synthetic.py
----------------------
Generates realistic synthetic MNREGA district-level data for Maharashtra.

Mimics the structure of real data available from:
- nregarep1.nic.in (MoRD official portal)
- dataful.in (district-wise persondays + expenditure)

Columns produced match what you'd get from real sources:
    state, district, financial_year,
    households_demanded, households_offered, households_availed,
    person_days, expenditure_lakhs, avg_wage_rate, works_completed

Design principles for realism:
    - Each district has a stable "base capacity" (some districts are
      structurally larger / more active than others)
    - Year-on-year growth follows real MNREGA trends (spike in 2020-21
      due to COVID reverse migration, slowdown in urban-adjacent districts)
    - Expenditure correlates with person_days but has noise (efficiency varies)
    - Wage rate increases over years (matches real wage revision schedule)
    - ~8% missing values injected randomly to simulate real data quality
"""

import numpy as np
import pandas as pd
import os

# ── Maharashtra districts (all 36) ───────────────────────────────────────────
MAHARASHTRA_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

YEARS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"
]

# Real MNREGA wage rates in Maharashtra (approx ₹/day by year)
WAGE_RATES = {
    "2014-15": 162, "2015-16": 174, "2016-17": 183, "2017-18": 194,
    "2018-19": 203, "2019-20": 213, "2020-21": 238, "2021-22": 256,
    "2022-23": 273, "2023-24": 289
}

# Year-level demand multipliers based on real MNREGA trends
# COVID year (2020-21) saw massive spike due to reverse migration
YEAR_MULTIPLIERS = {
    "2014-15": 0.85, "2015-16": 0.90, "2016-17": 0.92, "2017-18": 0.95,
    "2018-19": 1.00, "2019-20": 1.05, "2020-21": 1.45, "2021-22": 1.20,
    "2022-23": 1.10, "2023-24": 1.08
}

# District profile: (base_persondays_lakhs, efficiency_score, rural_weight)
# Urban/peri-urban districts have lower base; tribal/rural have higher
DISTRICT_PROFILES = {
    "Gadchiroli":      (18.5, 0.72, 0.95),
    "Nandurbar":       (16.2, 0.68, 0.93),
    "Yavatmal":        (15.8, 0.74, 0.91),
    "Amravati":        (14.3, 0.76, 0.88),
    "Chandrapur":      (13.9, 0.71, 0.87),
    "Washim":          (12.1, 0.73, 0.89),
    "Buldhana":        (11.8, 0.75, 0.86),
    "Beed":            (11.5, 0.70, 0.90),
    "Hingoli":         (10.9, 0.72, 0.88),
    "Osmanabad":       (10.7, 0.69, 0.87),
    "Latur":           (10.4, 0.71, 0.85),
    "Nanded":          (10.2, 0.73, 0.84),
    "Jalna":           (9.8,  0.74, 0.85),
    "Parbhani":        (9.5,  0.72, 0.84),
    "Akola":           (9.3,  0.75, 0.83),
    "Dhule":           (9.1,  0.70, 0.85),
    "Gondia":          (8.9,  0.76, 0.82),
    "Bhandara":        (8.6,  0.74, 0.81),
    "Wardha":          (8.3,  0.77, 0.80),
    "Ahmednagar":      (8.1,  0.78, 0.79),
    "Solapur":         (7.9,  0.76, 0.80),
    "Aurangabad":      (7.6,  0.79, 0.75),
    "Jalgaon":         (7.4,  0.77, 0.77),
    "Nashik":          (7.1,  0.80, 0.73),
    "Satara":          (6.8,  0.81, 0.74),
    "Sangli":          (6.5,  0.80, 0.73),
    "Kolhapur":        (6.2,  0.82, 0.71),
    "Palghar":         (6.0,  0.75, 0.78),
    "Nandurbar":       (5.8,  0.71, 0.82),
    "Ratnagiri":       (5.5,  0.79, 0.74),
    "Sindhudurg":      (5.1,  0.80, 0.72),
    "Raigad":          (4.8,  0.78, 0.68),
    "Pune":            (4.2,  0.83, 0.55),
    "Thane":           (3.5,  0.81, 0.45),
    "Mumbai Suburban": (1.2,  0.85, 0.15),
    "Mumbai City":     (0.4,  0.88, 0.05),
}


def generate(seed: int = 42, missing_rate: float = 0.08) -> pd.DataFrame:
    """
    Generate a synthetic MNREGA dataset for Maharashtra.

    Args:
        seed        : Random seed for reproducibility.
        missing_rate: Fraction of cells to nullify (simulates real data gaps).

    Returns:
        DataFrame with realistic MNREGA data.
    """
    rng = np.random.default_rng(seed)
    records = []

    for district in MAHARASHTRA_DISTRICTS:
        profile = DISTRICT_PROFILES.get(district, (7.0, 0.75, 0.70))
        base_pd, efficiency, rural_w = profile

        for year in YEARS:
            year_mult = YEAR_MULTIPLIERS[year]
            wage = WAGE_RATES[year]

            # ── Person days (in lakhs) ────────────────────────────────────
            noise = rng.normal(1.0, 0.07)
            person_days_lakhs = base_pd * year_mult * noise
            person_days_lakhs = max(person_days_lakhs, 0.1)

            # ── Households ───────────────────────────────────────────────
            # Avg ~45 days per household → households = person_days / 45
            hh_demanded = int(person_days_lakhs * 1e5 / 38 * rng.uniform(1.05, 1.15))
            hh_offered   = int(hh_demanded * rng.uniform(0.92, 0.99))
            hh_availed   = int(hh_offered  * rng.uniform(0.88, 0.97))

            # ── Expenditure (₹ lakhs) ────────────────────────────────────
            # Base = person_days * wage_rate, efficiency introduces noise
            base_expenditure = person_days_lakhs * 1e5 * wage / 1e5
            expenditure_lakhs = base_expenditure / efficiency * rng.uniform(0.93, 1.07)

            # ── Works completed ──────────────────────────────────────────
            works = int(person_days_lakhs * rng.uniform(18, 35))

            records.append({
                "state":                "Maharashtra",
                "district":             district,
                "financial_year":       year,
                "households_demanded":  hh_demanded,
                "households_offered":   hh_offered,
                "households_availed":   hh_availed,
                "person_days_lakhs":    round(person_days_lakhs, 3),
                "expenditure_lakhs":    round(expenditure_lakhs, 2),
                "avg_wage_rate":        wage,
                "works_completed":      works,
            })

    df = pd.DataFrame(records)

    # ── Inject realistic missing values ──────────────────────────────────────
    nullable_cols = [
        "households_demanded", "households_offered",
        "households_availed", "works_completed"
    ]
    for col in nullable_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    print(f"[generate] Created {len(df)} rows × {len(df.columns)} columns")
    print(f"[generate] Districts: {df['district'].nunique()} | Years: {df['financial_year'].nunique()}")
    print(f"[generate] Missing values injected: ~{missing_rate*100:.0f}% per nullable column")

    return df


def save(df: pd.DataFrame, path: str = "data/raw/mnrega_maharashtra_synthetic.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate] Saved → {path}")


if __name__ == "__main__":
    df = generate()
    save(df)
    print("\nSample:")
    print(df.head(6).to_string(index=False))
