import duckdb

FILE = r"D:\NutriRateAI\food.parquet"

db = duckdb.connect()
db.execute("SET threads = 8")

print("=" * 80)
print("NUTRIRATE AI - CLEANING IMPACT ANALYSIS")
print("=" * 80)


# ---------------------------------------------------------
# BUILD ML DATASET
# ---------------------------------------------------------

print("\nBuilding ML dataset...")

db.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_data AS

WITH unnested AS (

    SELECT
        code,
        nutriscore_grade,
        nutriscore_score,
        n.name AS nutrient_name,
        n."100g" AS value_100g

    FROM read_parquet('{FILE}') p

    LEFT JOIN UNNEST(p.nutriments) AS u(n)
        ON TRUE
)

SELECT

    code,

    UPPER(TRIM(nutriscore_grade)) AS grade,

    nutriscore_score,

    MAX(CASE
        WHEN nutrient_name = 'energy-kcal'
        THEN value_100g
    END) AS energy,

    MAX(CASE
        WHEN nutrient_name = 'fat'
        THEN value_100g
    END) AS fat,

    MAX(CASE
        WHEN nutrient_name = 'saturated-fat'
        THEN value_100g
    END) AS saturated_fat,

    MAX(CASE
        WHEN nutrient_name = 'carbohydrates'
        THEN value_100g
    END) AS carbohydrates,

    MAX(CASE
        WHEN nutrient_name = 'sugars'
        THEN value_100g
    END) AS sugars,

    MAX(CASE
        WHEN nutrient_name = 'proteins'
        THEN value_100g
    END) AS protein,

    MAX(CASE
        WHEN nutrient_name = 'fiber'
        THEN value_100g
    END) AS fiber,

    MAX(CASE
        WHEN nutrient_name = 'salt'
        THEN value_100g
    END) AS salt

FROM unnested

GROUP BY
    code,
    nutriscore_grade,
    nutriscore_score

HAVING
    nutriscore_score IS NOT NULL

    AND UPPER(TRIM(nutriscore_grade))
        IN ('A', 'B', 'C', 'D', 'E')

    AND energy IS NOT NULL
    AND fat IS NOT NULL
    AND saturated_fat IS NOT NULL
    AND carbohydrates IS NOT NULL
    AND sugars IS NOT NULL
    AND protein IS NOT NULL
    AND fiber IS NOT NULL
    AND salt IS NOT NULL
""")


total = db.execute("""
SELECT COUNT(*)
FROM ml_data
""").fetchone()[0]

print(f"Initial ML rows: {total:,}")


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("DUPLICATES")
print("-" * 80)

duplicates = db.execute("""
SELECT COUNT(*)
FROM (
    SELECT code
    FROM ml_data
    GROUP BY code
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

duplicate_rows = db.execute("""
SELECT
    COUNT(*) - COUNT(DISTINCT code)
FROM ml_data
""").fetchone()[0]

print(f"Products with duplicate codes: {duplicates:,}")
print(f"Duplicate rows beyond first:    {duplicate_rows:,}")


# ---------------------------------------------------------
# INDIVIDUAL NEGATIVE CHECKS
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("NEGATIVE VALUES")
print("-" * 80)

negative_rules = [
    ("energy", "Energy"),
    ("fat", "Fat"),
    ("saturated_fat", "Saturated fat"),
    ("carbohydrates", "Carbohydrates"),
    ("sugars", "Sugars"),
    ("protein", "Protein"),
    ("fiber", "Fiber"),
    ("salt", "Salt")
]

for column, name in negative_rules:

    count = db.execute(f"""
        SELECT COUNT(*)
        FROM ml_data
        WHERE {column} < 0
    """).fetchone()[0]

    print(f"{name:20s}: {count:,}")


# ---------------------------------------------------------
# SENSIBLE NUTRITION BOUNDS
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("PHYSICALLY SENSIBLE BOUNDS")
print("-" * 80)

print("""
The following limits are used only for impact analysis:

Energy          : 0 - 1000 kcal / 100g
Fat             : 0 - 100 g / 100g
Saturated fat   : 0 - 100 g / 100g
Carbohydrates   : 0 - 100 g / 100g
Sugars          : 0 - 100 g / 100g
Protein         : 0 - 100 g / 100g
Fiber           : 0 - 100 g / 100g
Salt            : 0 - 100 g / 100g
""")


bounds = [
    ("energy", "Energy", 1000),
    ("fat", "Fat", 100),
    ("saturated_fat", "Saturated fat", 100),
    ("carbohydrates", "Carbohydrates", 100),
    ("sugars", "Sugars", 100),
    ("protein", "Protein", 100),
    ("fiber", "Fiber", 100),
    ("salt", "Salt", 100)
]


for column, name, maximum in bounds:

    count = db.execute(f"""
        SELECT COUNT(*)
        FROM ml_data
        WHERE {column} < 0
           OR {column} > {maximum}
    """).fetchone()[0]

    percentage = count / total * 100

    print(
        f"{name:20s}: "
        f"{count:,} "
        f"({percentage:.3f}%)"
    )


# ---------------------------------------------------------
# ROWS FAILING ANY RULE
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("COMBINED CLEANING IMPACT")
print("-" * 80)

invalid_query = """
SELECT COUNT(*)
FROM ml_data
WHERE

    -- Energy
    energy < 0 OR energy > 1000

    OR

    -- Fat
    fat < 0 OR fat > 100

    OR

    -- Saturated fat
    saturated_fat < 0 OR saturated_fat > 100

    OR

    -- Carbohydrates
    carbohydrates < 0 OR carbohydrates > 100

    OR

    -- Sugars
    sugars < 0 OR sugars > 100

    OR

    -- Protein
    protein < 0 OR protein > 100

    OR

    -- Fiber
    fiber < 0 OR fiber > 100

    OR

    -- Salt
    salt < 0 OR salt > 100
"""

invalid_rows = db.execute(invalid_query).fetchone()[0]

remaining = total - invalid_rows

print(f"Initial rows:       {total:,}")
print(f"Rows failing rules: {invalid_rows:,}")
print(f"Rows remaining:     {remaining:,}")
print(f"Rows retained:      {remaining / total * 100:.3f}%")
print(f"Rows removed:       {invalid_rows / total * 100:.3f}%")


# ---------------------------------------------------------
# CLASS DISTRIBUTION BEFORE CLEANING
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("CLASS DISTRIBUTION - BEFORE CLEANING")
print("-" * 80)

before = db.execute("""
SELECT
    grade,
    COUNT(*)
FROM ml_data
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in before:
    print(f"{grade}: {count:,}")


# ---------------------------------------------------------
# CLASS DISTRIBUTION AFTER CLEANING
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("CLASS DISTRIBUTION - AFTER CLEANING")
print("-" * 80)

after = db.execute("""
SELECT
    grade,
    COUNT(*)
FROM ml_data
WHERE
    energy BETWEEN 0 AND 1000
    AND fat BETWEEN 0 AND 100
    AND saturated_fat BETWEEN 0 AND 100
    AND carbohydrates BETWEEN 0 AND 100
    AND sugars BETWEEN 0 AND 100
    AND protein BETWEEN 0 AND 100
    AND fiber BETWEEN 0 AND 100
    AND salt BETWEEN 0 AND 100
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in after:
    percentage = count / remaining * 100
    print(
        f"{grade}: {count:,} "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# SCORE DISTRIBUTION AFTER CLEANING
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("NUTRI-SCORE SCORE AFTER CLEANING")
print("-" * 80)

score = db.execute("""
SELECT
    MIN(nutriscore_score),
    MAX(nutriscore_score),
    AVG(nutriscore_score),
    MEDIAN(nutriscore_score)
FROM ml_data
WHERE
    energy BETWEEN 0 AND 1000
    AND fat BETWEEN 0 AND 100
    AND saturated_fat BETWEEN 0 AND 100
    AND carbohydrates BETWEEN 0 AND 100
    AND sugars BETWEEN 0 AND 100
    AND protein BETWEEN 0 AND 100
    AND fiber BETWEEN 0 AND 100
    AND salt BETWEEN 0 AND 100
""").fetchone()

print(f"Minimum score: {score[0]}")
print(f"Maximum score: {score[1]}")
print(f"Mean score:    {score[2]:.4f}")
print(f"Median score:  {score[3]:.4f}")


# ---------------------------------------------------------
# FINAL DECISION SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CLEANING IMPACT COMPLETE")
print("=" * 80)

print(f"""
Original ML dataset : {total:,}
Rows failing rules  : {invalid_rows:,}
Rows retained       : {remaining:,}
Retention rate      : {remaining / total * 100:.3f}%

NO DATA HAS BEEN MODIFIED.
NO ROWS HAVE BEEN DELETED.
""")

db.close()