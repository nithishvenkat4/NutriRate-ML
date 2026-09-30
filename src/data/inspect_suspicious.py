import duckdb

FILE = r"D:\NutriRateAI\nutrirate_ml_clean.parquet"

db = duckdb.connect()

print("=" * 80)
print("NUTRIRATE AI - SUSPICIOUS RECORD INSPECTION")
print("=" * 80)


# ---------------------------------------------------------
# ALL-ZERO RECORDS
# ---------------------------------------------------------

print("\nALL-ZERO RECORDS")
print("-" * 80)

zeros = db.execute(f"""
SELECT
    code,
    grade,
    nutriscore_score,
    energy,
    fat,
    saturated_fat,
    carbohydrates,
    sugars,
    protein,
    fiber,
    salt
FROM read_parquet('{FILE}')
WHERE energy = 0
  AND fat = 0
  AND saturated_fat = 0
  AND carbohydrates = 0
  AND sugars = 0
  AND protein = 0
  AND fiber = 0
  AND salt = 0
LIMIT 30
""").fetchall()

for row in zeros:
    print(row)


# ---------------------------------------------------------
# ALL-ONES RECORDS
# ---------------------------------------------------------

print("\n\nALL-ONES RECORDS")
print("-" * 80)

ones = db.execute(f"""
SELECT
    code,
    grade,
    nutriscore_score,
    energy,
    fat,
    saturated_fat,
    carbohydrates,
    sugars,
    protein,
    fiber,
    salt
FROM read_parquet('{FILE}')
WHERE energy = 1
  AND fat = 1
  AND saturated_fat = 1
  AND carbohydrates = 1
  AND sugars = 1
  AND protein = 1
  AND fiber = 1
  AND salt = 1
LIMIT 30
""").fetchall()

for row in ones:
    print(row)


# ---------------------------------------------------------
# ZERO RECORD GRADE DISTRIBUTION
# ---------------------------------------------------------

print("\n\nALL-ZERO GRADE DISTRIBUTION")
print("-" * 80)

zero_grades = db.execute(f"""
SELECT
    grade,
    COUNT(*) AS count
FROM read_parquet('{FILE}')
WHERE energy = 0
  AND fat = 0
  AND saturated_fat = 0
  AND carbohydrates = 0
  AND sugars = 0
  AND protein = 0
  AND fiber = 0
  AND salt = 0
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in zero_grades:
    print(f"{grade}: {count:,}")


# ---------------------------------------------------------
# ONES GRADE DISTRIBUTION
# ---------------------------------------------------------

print("\n\nALL-ONES GRADE DISTRIBUTION")
print("-" * 80)

one_grades = db.execute(f"""
SELECT
    grade,
    COUNT(*) AS count
FROM read_parquet('{FILE}')
WHERE energy = 1
  AND fat = 1
  AND saturated_fat = 1
  AND carbohydrates = 1
  AND sugars = 1
  AND protein = 1
  AND fiber = 1
  AND salt = 1
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in one_grades:
    print(f"{grade}: {count:,}")


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)