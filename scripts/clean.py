import pandas as pd
import numpy as np
import os

# ════════════════════════════════════════════
#  SPOTIFY USER BEHAVIOR — DATA CLEANING
#  Problem: Identify drivers of Churn,
#           Ad Conversion & Engagement
# ════════════════════════════════════════════

# ── LOAD ─────────────────────────────────────
df = pd.read_csv("data/raw/spotify_raw.csv")
print(f"✅ Loaded → {df.shape[0]} rows, {df.shape[1]} columns")


# ════════════════════════════════════════════
#  PILLAR 1 — CHURN
#  Sheets: Churn by Country, Churn by Age Group,
#          Inactivity Duration
# ════════════════════════════════════════════

# Age groups — needed for "Churn by Age Group" sheet
# Bins divide users into 5 career-life segments
bins   = [15, 24, 34, 44, 54, 61]
labels = ["16-24", "25-34", "35-44", "45-54", "55-60"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
# Interview point: age_group lets Tableau group a continuous
# variable into business-meaningful segments without hardcoding

# Churn flag — 1 if user is Inactive, 0 if Active
# Used for calculating churn rate as a % in Tableau
df["churn_flag"] = (df["subscription_status"] == "Inactive").astype(int)
# Interview point: converting string → 0/1 lets us use AVG()
# in Tableau to get churn rate directly (e.g. AVG = 0.32 = 32%)

# Inactivity bucket — groups months_inactive into bands
# Needed for "Inactivity Duration" histogram sheet
df["inactivity_bucket"] = pd.cut(
    df["months_inactive"],
    bins=[-1, 0, 2, 5, 10, 18],
    labels=["0 months", "1-2 months", "3-5 months", "6-10 months", "10+ months"]
)
# Interview point: raw months_inactive has 18 unique values —
# binning into 5 buckets makes the histogram readable


# ════════════════════════════════════════════
#  PILLAR 2 — AD CONVERSION
#  Sheets: Ad Conversion by Plan,
#          Ad Impact on Engagement
# ════════════════════════════════════════════

# Convert Yes/No → 1/0 for both ad columns
# AVG() of these in Tableau gives conversion rate as a decimal
df["ad_interaction_flag"] = (df["ad_interaction"] == "Yes").astype(int)
# Interview point: 1 = user saw and interacted with an ad

df["ad_conversion_flag"] = (df["ad_conversion_to_subscription"] == "Yes").astype(int)
# Interview point: 1 = user converted from Free to paid after ad
# AVG(ad_conversion_flag) grouped by subscription_type gives
# conversion rate per plan — core of Sheet 4


# ════════════════════════════════════════════
#  PILLAR 3 — ENGAGEMENT
#  Sheets: Listening Hours by Age, Skip Rate by Genre,
#          Playlist Engagement by Plan
# ════════════════════════════════════════════

# Engagement score — composite metric combining 3 signals
# Higher score = more engaged user = lower churn risk
df["engagement_score"] = (
    (df["avg_listening_hours_per_week"] * 0.5) +
    (df["playlists_created"]            * 0.3) -
    (df["avg_skips_per_day"]            * 0.2)
).round(2)
# Interview point: weighted formula because listening hours
# matter more than playlists, and skips are a negative signal

# Skip rate category — for color-coding in Tableau
df["skip_category"] = pd.cut(
    df["avg_skips_per_day"],
    bins=[0, 5, 12, 25],
    labels=["Low", "Medium", "High"]
)
# Interview point: raw skip number is hard to read on a chart —
# Low/Medium/High makes it immediately interpretable

# Rename long column — cleaner in Tableau field list
df.rename(columns={
    "music_suggestion_rating_1_to_5": "music_suggestion_rating"
}, inplace=True)


# ════════════════════════════════════════════
#  DATE ENGINEERING — for Signups Over Time
# ════════════════════════════════════════════

df["signup_date"]      = pd.to_datetime(df["signup_date"])
df["signup_year"]      = df["signup_date"].dt.year
df["signup_month"]     = df["signup_date"].dt.month_name()
df["signup_yearmonth"] = df["signup_date"].dt.to_period("M").astype(str)
# Interview point: Tableau needs date broken into parts
# to build time-series line charts properly


# ════════════════════════════════════════════
#  QUALITY CHECKS
# ════════════════════════════════════════════

print("\n── Null Check ──")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "No nulls found ✅")

print("\n── Duplicate Check ──")
dupes = df.duplicated().sum()
print(f"Duplicates found: {dupes}")
df.drop_duplicates(inplace=True)

print("\n── Shape after cleaning ──")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# ════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════

os.makedirs("data/cleaned", exist_ok=True)
df.to_csv("data/cleaned/spotify_cleaned.csv", index=False)
print("\n✅ Saved → data/cleaned/spotify_cleaned.csv")
print(f"New columns added: age_group, churn_flag, inactivity_bucket,")
print(f"                   ad_interaction_flag, ad_conversion_flag,")
print(f"                   engagement_score, skip_category,")
print(f"                   signup_year, signup_month, signup_yearmonth")