import duckdb

FILE = r"D:\NutriRateAI\nutrirate_ml_clean.parquet"

db = duckdb.connect()

print("=" * 80)
print("NUTRIRATE AI - ML DATASET QUALITY ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------
# BASIC
# ---------------------------------------------------------

basic = db.execute(f"""
SELECT
    COUNT(*) AS total,
    COUNT(DISTINCT code) AS unique_codes,
    COUNT(*) FILTER (WHERE grade IS NULL) AS missing_grade,
    COUNT(*) FILTER (WHERE nutriscore_score IS NULL) AS missing_score
FROM read_parquet('{FILE}')
""").fetchone()

print("\nBASIC INFORMATION")
print("-" * 50)
print(f"Total rows:        {basic[0]:,}")
print(f"Unique codes:      {basic[1]:,}")
print(f"Missing grade:     {basic[2]:,}")
print(f"Missing score:     {basic[3]:,}")


# ---------------------------------------------------------
# GRADE DISTRIBUTION
# ---------------------------------------------------------

print("\nGRADE DISTRIBUTION")
print("-" * 50)

grades = db.execute(f"""
SELECT
    grade,
    COUNT(*) AS count
FROM read_parquet('{FILE}')
GROUP BY grade
ORDER BY grade
""").fetchall()

total = basic[0]

for grade, count in grades:
    print(
        f"{grade}: "
        f"{count:,} "
        f"({count / total * 100:.2f}%)"
    )


# ---------------------------------------------------------
# SCORE STATISTICS
# ---------------------------------------------------------

print("\nNUTRI-SCORE STATISTICS")
print("-" * 50)

score = db.execute(f"""
SELECT
    MIN(nutriscore_score),
    MAX(nutriscore_score),
    AVG(nutriscore_score),
    MEDIAN(nutriscore_score),
    STDDEV_SAMP(nutriscore_score),

    QUANTILE_CONT(nutriscore_score, 0.01),
    QUANTILE_CONT(nutriscore_score, 0.25),
    QUANTILE_CONT(nutriscore_score, 0.50),
    QUANTILE_CONT(nutriscore_score, 0.75),
    QUANTILE_CONT(nutriscore_score, 0.99)

FROM read_parquet('{FILE}')
""").fetchone()

print(f"Minimum:       {score[0]}")
print(f"Maximum:       {score[1]}")
print(f"Mean:          {score[2]:.2f}")
print(f"Median:        {score[3]:.2f}")
print(f"Std deviation: {score[4]:.2f}")
print(f"1st percentile:  {score[5]:.2f}")
print(f"25th percentile: {score[6]:.2f}")
print(f"50th percentile: {score[7]:.2f}")
print(f"75th percentile: {score[8]:.2f}")
print(f"99th percentile: {score[9]:.2f}")


# ---------------------------------------------------------
# NUTRITION STATISTICS
# ---------------------------------------------------------

features = [
    "energy",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "protein",
    "fiber",
    "salt"
]

print("\nNUTRITION FEATURE STATISTICS")
print("-" * 80)

for feature in features:

    result = db.execute(f"""
    SELECT
        MIN({feature}),
        MAX({feature}),
        AVG({feature}),
        MEDIAN({feature}),
        STDDEV_SAMP({feature}),
        QUANTILE_CONT({feature}, 0.01),
        QUANTILE_CONT({feature}, 0.99)
    FROM read_parquet('{FILE}')
    """).fetchone()

    print(f"\n{feature}")
    print(f"  Min:     {result[0]:.4f}")
    print(f"  Max:     {result[1]:.4f}")
    print(f"  Mean:    {result[2]:.4f}")
    print(f"  Median:  {result[3]:.4f}")
    print(f"  Std:     {result[4]:.4f}")
    print(f"  P1:      {result[5]:.4f}")
    print(f"  P99:     {result[6]:.4f}")


# ---------------------------------------------------------
# NEGATIVE VALUES
# ---------------------------------------------------------

print("\nNEGATIVE VALUE CHECK")
print("-" * 50)

for feature in features:

    count = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
    WHERE {feature} < 0
    """).fetchone()[0]

    print(f"{feature:20s}: {count:,}")


# ---------------------------------------------------------
# ALL FEATURES = 1
# ---------------------------------------------------------

print("\nSUSPICIOUS ALL-ONES RECORDS")
print("-" * 50)

all_one = db.execute(f"""
SELECT COUNT(*)
FROM read_parquet('{FILE}')
WHERE energy = 1
  AND fat = 1
  AND saturated_fat = 1
  AND carbohydrates = 1
  AND sugars = 1
  AND protein = 1
  AND fiber = 1
  AND salt = 1
""").fetchone()[0]

print(f"All nutritional features = 1: {all_one:,}")


# ---------------------------------------------------------
# ALL FEATURES = 0
# ---------------------------------------------------------

print("\nALL-ZERO RECORDS")
print("-" * 50)

all_zero = db.execute(f"""
SELECT COUNT(*)
FROM read_parquet('{FILE}')
WHERE energy = 0
  AND fat = 0
  AND saturated_fat = 0
  AND carbohydrates = 0
  AND sugars = 0
  AND protein = 0
  AND fiber = 0
  AND salt = 0
""").fetchone()[0]

print(f"All nutritional features = 0: {all_zero:,}")


# ---------------------------------------------------------
# CORRELATION WITH TARGET
# ---------------------------------------------------------

print("\nCORRELATION WITH NUTRI-SCORE")
print("-" * 50)

for feature in features:

    corr = db.execute(f"""
    SELECT CORR({feature}, nutriscore_score)
    FROM read_parquet('{FILE}')
    """).fetchone()[0]

    print(f"{feature:20s}: {corr:.4f}")


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)