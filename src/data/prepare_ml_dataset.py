import duckdb
import os

INPUT_FILE = r"D:\NutriRateAI\food.parquet"
OUTPUT_FILE = r"D:\NutriRateAI\nutrirate_ml_clean.parquet"

db = duckdb.connect()
db.execute("SET threads = 8")

print("=" * 80)
print("NUTRIRATE AI - PREPARE CLEAN ML DATASET")
print("=" * 80)


# ---------------------------------------------------------
# BUILD CLEAN DATASET
# ---------------------------------------------------------

print("\nExtracting and cleaning nutrition data...")

query = f"""

WITH unnested AS (

    SELECT
        code,
        nutriscore_grade,
        nutriscore_score,

        n.name AS nutrient_name,
        n."100g" AS value_100g

    FROM read_parquet('{INPUT_FILE}') p

    LEFT JOIN UNNEST(p.nutriments) AS u(n)
        ON TRUE
),

product_nutrition AS (

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

),

ranked AS (

    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY code
            ORDER BY code
        ) AS row_num

    FROM product_nutrition

    WHERE
        grade IN ('A', 'B', 'C', 'D', 'E')

        AND nutriscore_score IS NOT NULL

        AND energy BETWEEN 0 AND 1000
        AND fat BETWEEN 0 AND 100
        AND saturated_fat BETWEEN 0 AND 100
        AND carbohydrates BETWEEN 0 AND 100
        AND sugars BETWEEN 0 AND 100
        AND protein BETWEEN 0 AND 100
        AND fiber BETWEEN 0 AND 100
        AND salt BETWEEN 0 AND 100
)

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

FROM ranked

WHERE row_num = 1

ORDER BY code

"""


# ---------------------------------------------------------
# WRITE PARQUET
# ---------------------------------------------------------

print("\nWriting clean dataset...")

db.execute(f"""
COPY (
    {query}
)
TO '{OUTPUT_FILE}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ---------------------------------------------------------
# VERIFY
# ---------------------------------------------------------

print("\nVerifying output...")

result = db.execute(f"""
SELECT COUNT(*)
FROM read_parquet('{OUTPUT_FILE}')
""").fetchone()[0]

print(f"Clean ML rows: {result:,}")


# ---------------------------------------------------------
# VERIFY DUPLICATES
# ---------------------------------------------------------

duplicates = db.execute(f"""
SELECT
    COUNT(*) - COUNT(DISTINCT code)
FROM read_parquet('{OUTPUT_FILE}')
""").fetchone()[0]

print(f"Duplicate codes: {duplicates:,}")


# ---------------------------------------------------------
# VERIFY INVALID VALUES
# ---------------------------------------------------------

invalid = db.execute(f"""
SELECT COUNT(*)
FROM read_parquet('{OUTPUT_FILE}')
WHERE
    energy < 0 OR energy > 1000
    OR fat < 0 OR fat > 100
    OR saturated_fat < 0 OR saturated_fat > 100
    OR carbohydrates < 0 OR carbohydrates > 100
    OR sugars < 0 OR sugars > 100
    OR protein < 0 OR protein > 100
    OR fiber < 0 OR fiber > 100
    OR salt < 0 OR salt > 100
""").fetchone()[0]

print(f"Invalid nutrition rows: {invalid}")


# ---------------------------------------------------------
# FILE SIZE
# ---------------------------------------------------------

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

print(f"Output file size: {file_size_mb:.2f} MB")


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nFinal class distribution:")

classes = db.execute(f"""
SELECT
    grade,
    COUNT(*) AS count
FROM read_parquet('{OUTPUT_FILE}')
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in classes:

    percentage = count / result * 100

    print(
        f"  {grade}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CLEAN ML DATASET READY")
print("=" * 80)

print(f"""
Input:
  {INPUT_FILE}

Output:
  {OUTPUT_FILE}

Rows:
  {result:,}

Duplicates:
  {duplicates:,}

Invalid rows:
  {invalid:,}
""")

db.close()