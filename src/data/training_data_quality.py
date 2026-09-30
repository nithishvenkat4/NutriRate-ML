import duckdb

FILE = r"D:\NutriRateAI\food.parquet"

db = duckdb.connect()
db.execute("SET threads = 8")

print("=" * 75)
print("NUTRIRATE AI - TRAINING DATA QUALITY ANALYSIS")
print("=" * 75)


# ---------------------------------------------------------
# CREATE ML DATASET
# ---------------------------------------------------------

print("\nBuilding training dataset...")

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


count = db.execute("""
SELECT COUNT(*)
FROM ml_data
""").fetchone()[0]

print(f"Training rows: {count:,}")


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("CLASS DISTRIBUTION")
print("-" * 75)

classes = db.execute("""
SELECT
    grade,
    COUNT(*) AS count
FROM ml_data
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, value in classes:

    percentage = value / count * 100

    print(
        f"{grade}: "
        f"{value:,} "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# NUTRITION STATISTICS
# ---------------------------------------------------------

features = [
    ("energy", "Energy"),
    ("fat", "Fat"),
    ("saturated_fat", "Saturated Fat"),
    ("carbohydrates", "Carbohydrates"),
    ("sugars", "Sugars"),
    ("protein", "Protein"),
    ("fiber", "Fiber"),
    ("salt", "Salt")
]


print("\n" + "-" * 75)
print("NUTRITION STATISTICS")
print("-" * 75)


for column, name in features:

    result = db.execute(f"""
        SELECT
            MIN({column}),
            MAX({column}),
            AVG({column}),
            MEDIAN({column}),
            STDDEV({column})
        FROM ml_data
    """).fetchone()

    minimum, maximum, average, median, stddev = result

    print(f"\n{name}")
    print(f"  Min:    {minimum}")
    print(f"  Max:    {maximum}")
    print(f"  Mean:   {average:.4f}")
    print(f"  Median: {median:.4f}")
    print(f"  Std:    {stddev:.4f}")


# ---------------------------------------------------------
# NEGATIVE VALUES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("NEGATIVE VALUE CHECK")
print("-" * 75)


for column, name in features:

    negative = db.execute(f"""
        SELECT COUNT(*)
        FROM ml_data
        WHERE {column} < 0
    """).fetchone()[0]

    print(f"{name:20s}: {negative:,}")


# ---------------------------------------------------------
# EXTREME VALUE CHECK
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("EXTREME VALUE CHECK")
print("-" * 75)

checks = [
    ("energy", 1000),
    ("fat", 100),
    ("saturated_fat", 100),
    ("carbohydrates", 100),
    ("sugars", 100),
    ("protein", 100),
    ("fiber", 100),
    ("salt", 50)
]


for column, limit in checks:

    result = db.execute(f"""
        SELECT COUNT(*)
        FROM ml_data
        WHERE {column} > {limit}
    """).fetchone()[0]

    print(
        f"{column:20s}: "
        f"{result:,} > {limit}"
    )


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("DUPLICATE CHECK")
print("-" * 75)

duplicate_codes = db.execute("""
SELECT
    COUNT(*) - COUNT(DISTINCT code)
FROM ml_data
""").fetchone()[0]

print(f"Duplicate product codes: {duplicate_codes:,}")


# ---------------------------------------------------------
# NUTRI-SCORE SCORE RANGE
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("NUTRI-SCORE SCORE RANGE")
print("-" * 75)

score_stats = db.execute("""
SELECT
    MIN(nutriscore_score),
    MAX(nutriscore_score),
    AVG(nutriscore_score),
    MEDIAN(nutriscore_score)
FROM ml_data
""").fetchone()

print(f"Minimum score: {score_stats[0]}")
print(f"Maximum score: {score_stats[1]}")
print(f"Mean score:    {score_stats[2]:.4f}")
print(f"Median score:  {score_stats[3]:.4f}")


# ---------------------------------------------------------
# SCORE BY CLASS
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("NUTRI-SCORE SCORE BY CLASS")
print("-" * 75)

score_by_class = db.execute("""
SELECT
    grade,
    COUNT(*) AS count,
    AVG(nutriscore_score) AS average_score,
    MIN(nutriscore_score) AS minimum_score,
    MAX(nutriscore_score) AS maximum_score

FROM ml_data

GROUP BY grade

ORDER BY grade
""").fetchall()


for row in score_by_class:

    grade, count_class, avg_score, min_score, max_score = row

    print(
        f"{grade}: "
        f"{count_class:,} products | "
        f"avg={avg_score:.2f} | "
        f"range={min_score} to {max_score}"
    )


print("\n" + "=" * 75)
print("QUALITY ANALYSIS COMPLETE")
print("=" * 75)

db.close()