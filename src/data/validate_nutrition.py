import duckdb

FILE = r"D:\NutriRateAI\nutrirate_ml_clean.parquet"

db = duckdb.connect()

print("=" * 80)
print("NUTRIRATE AI - NUTRITION CONSISTENCY CHECK")
print("=" * 80)

base = f"read_parquet('{FILE}')"

# ---------------------------------------------------------
# REMOVE OBVIOUS PLACEHOLDERS FOR THIS ANALYSIS
# ---------------------------------------------------------

valid = f"""
SELECT *
FROM {base}
WHERE NOT (
    energy = 0
    AND fat = 0
    AND saturated_fat = 0
    AND carbohydrates = 0
    AND sugars = 0
    AND protein = 0
    AND fiber = 0
    AND salt = 0
)
AND NOT (
    energy = 1
    AND fat = 1
    AND saturated_fat = 1
    AND carbohydrates = 1
    AND sugars = 1
    AND protein = 1
    AND fiber = 1
    AND salt = 1
)
"""

# ---------------------------------------------------------
# TOTAL VALID BASE
# ---------------------------------------------------------

total = db.execute(f"""
SELECT COUNT(*)
FROM ({valid})
""").fetchone()[0]

print(f"\nRecords after placeholder removal: {total:,}")


# ---------------------------------------------------------
# CONSISTENCY CHECKS
# ---------------------------------------------------------

checks = {

    "Sugar > carbohydrates":
    "sugars > carbohydrates",

    "Saturated fat > total fat":
    "saturated_fat > fat",

    "Fiber > carbohydrates":
    "fiber > carbohydrates",

    "Protein > 100g":
    "protein > 100",

    "Fat > 100g":
    "fat > 100",

    "Saturated fat > 100g":
    "saturated_fat > 100",

    "Carbohydrates > 100g":
    "carbohydrates > 100",

    "Sugars > 100g":
    "sugars > 100",

    "Fiber > 100g":
    "fiber > 100",

    "Salt > 100g":
    "salt > 100",

    "Energy > 1000 kcal":
    "energy > 1000",

    "Negative energy":
    "energy < 0"
}

print("\nCONSISTENCY CHECKS")
print("-" * 80)

for name, condition in checks.items():

    count = db.execute(f"""
    SELECT COUNT(*)
    FROM ({valid})
    WHERE {condition}
    """).fetchone()[0]

    percentage = count / total * 100

    print(
        f"{name:35s}: "
        f"{count:,} ({percentage:.3f}%)"
    )


# ---------------------------------------------------------
# SHOW EXTREME RECORDS
# ---------------------------------------------------------

print("\nTOP 10 HIGHEST ENERGY RECORDS")
print("-" * 80)

rows = db.execute(f"""
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
FROM ({valid})
ORDER BY energy DESC
LIMIT 10
""").fetchall()

for row in rows:
    print(row)


# ---------------------------------------------------------
# SCORE-GRADE CONSISTENCY
# ---------------------------------------------------------

print("\nNUTRI-SCORE GRADE / SCORE DISTRIBUTION")
print("-" * 80)

rows = db.execute(f"""
SELECT
    grade,
    MIN(nutriscore_score) AS min_score,
    MAX(nutriscore_score) AS max_score,
    AVG(nutriscore_score) AS avg_score,
    COUNT(*) AS count
FROM ({valid})
GROUP BY grade
ORDER BY grade
""").fetchall()

for row in rows:
    print(
        f"Grade {row[0]} | "
        f"Min: {row[1]} | "
        f"Max: {row[2]} | "
        f"Mean: {row[3]:.2f} | "
        f"Count: {row[4]:,}"
    )


print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)