import duckdb

FILE = r"D:\NutriRateAI\food.parquet"

db = duckdb.connect()

print("=" * 70)
print("NUTRIRATE AI - CORRECT DATASET ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------

print("\nReading basic statistics...")

basic = db.execute(f"""
SELECT
    COUNT(*) AS total_products,

    COUNT(*) FILTER (
        WHERE nutriscore_score IS NOT NULL
    ) AS with_nutriscore,

    COUNT(*) FILTER (
        WHERE ingredients IS NOT NULL
        AND LENGTH(TRIM(ingredients)) > 0
    ) AS with_ingredients,

    COUNT(*) FILTER (
        WHERE images IS NOT NULL
        AND ARRAY_LENGTH(images) > 0
    ) AS with_images,

    COUNT(*) FILTER (
        WHERE nutriments IS NOT NULL
        AND no_nutrition_data = FALSE
    ) AS with_nutrition

FROM read_parquet('{FILE}')
""").fetchone()

print(f"Total products:       {basic[0]:,}")
print(f"With Nutri-Score:     {basic[1]:,}")
print(f"With ingredients:     {basic[2]:,}")
print(f"With images:          {basic[3]:,}")
print(f"With nutrition data:  {basic[4]:,}")


# ---------------------------------------------------------
# NUTRI-SCORE DISTRIBUTION
# ---------------------------------------------------------

print("\nNutri-Score distribution:")

grades = db.execute(f"""
SELECT
    UPPER(TRIM(nutriscore_grade)) AS grade,
    COUNT(*) AS count
FROM read_parquet('{FILE}')
WHERE LOWER(TRIM(nutriscore_grade))
      IN ('a', 'b', 'c', 'd', 'e')
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in grades:
    print(f"  {grade}: {count:,}")


# ---------------------------------------------------------
# CREATE PRODUCT-LEVEL NUTRITION VIEW
# ---------------------------------------------------------

print("\nExtracting product-level nutrition values...")

query = f"""
CREATE OR REPLACE TEMP VIEW nutrition_data AS

SELECT

    code,

    nutriscore_grade,

    nutriscore_score,

    -- Energy
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'energy-kcal'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS energy_kcal_100g,

    -- Fat
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'fat'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS fat_100g,

    -- Saturated fat
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'saturated-fat'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS saturated_fat_100g,

    -- Carbohydrates
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'carbohydrates'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS carbohydrates_100g,

    -- Sugars
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'sugars'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS sugars_100g,

    -- Protein
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'proteins'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS proteins_100g,

    -- Fiber
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'fiber'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS fiber_100g,

    -- Salt
    (
        SELECT n."100g"
        FROM UNNEST(nutriments) AS u(n)
        WHERE n.name = 'salt'
          AND n."100g" IS NOT NULL
        LIMIT 1
    ) AS salt_100g

FROM read_parquet('{FILE}')
"""

db.execute(query)


# ---------------------------------------------------------
# NUTRIENT AVAILABILITY
# ---------------------------------------------------------

print("\nCore nutrient availability:")

nutrients = [
    ("energy_kcal_100g", "Energy"),
    ("fat_100g", "Fat"),
    ("saturated_fat_100g", "Saturated fat"),
    ("carbohydrates_100g", "Carbohydrates"),
    ("sugars_100g", "Sugars"),
    ("proteins_100g", "Protein"),
    ("fiber_100g", "Fiber"),
    ("salt_100g", "Salt")
]

for column, name in nutrients:

    count = db.execute(f"""
        SELECT COUNT(*)
        FROM nutrition_data
        WHERE {column} IS NOT NULL
    """).fetchone()[0]

    percentage = count / basic[0] * 100

    print(
        f"  {name:20s}: "
        f"{count:,} ({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# ALL 8 NUTRIENTS
# ---------------------------------------------------------

print("\nProducts with all 8 core nutrients:")

complete = db.execute("""
SELECT COUNT(*)
FROM nutrition_data
WHERE energy_kcal_100g IS NOT NULL
  AND fat_100g IS NOT NULL
  AND saturated_fat_100g IS NOT NULL
  AND carbohydrates_100g IS NOT NULL
  AND sugars_100g IS NOT NULL
  AND proteins_100g IS NOT NULL
  AND fiber_100g IS NOT NULL
  AND salt_100g IS NOT NULL
""").fetchone()[0]

print(f"  {complete:,}")


# ---------------------------------------------------------
# COMPLETE + NUTRI-SCORE
# ---------------------------------------------------------

print("\nProducts suitable for supervised ANN training:")

training = db.execute("""
SELECT COUNT(*)
FROM nutrition_data
WHERE nutriscore_score IS NOT NULL
  AND LOWER(TRIM(nutriscore_grade))
      IN ('a', 'b', 'c', 'd', 'e')
  AND energy_kcal_100g IS NOT NULL
  AND fat_100g IS NOT NULL
  AND saturated_fat_100g IS NOT NULL
  AND carbohydrates_100g IS NOT NULL
  AND sugars_100g IS NOT NULL
  AND proteins_100g IS NOT NULL
  AND fiber_100g IS NOT NULL
  AND salt_100g IS NOT NULL
""").fetchone()[0]

print(f"  {training:,}")


# ---------------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\nTraining target distribution:")

target = db.execute("""
SELECT
    UPPER(TRIM(nutriscore_grade)) AS grade,
    COUNT(*) AS count
FROM nutrition_data
WHERE nutriscore_score IS NOT NULL
  AND LOWER(TRIM(nutriscore_grade))
      IN ('a', 'b', 'c', 'd', 'e')
  AND energy_kcal_100g IS NOT NULL
  AND fat_100g IS NOT NULL
  AND saturated_fat_100g IS NOT NULL
  AND carbohydrates_100g IS NOT NULL
  AND sugars_100g IS NOT NULL
  AND proteins_100g IS NOT NULL
  AND fiber_100g IS NOT NULL
  AND salt_100g IS NOT NULL
GROUP BY grade
ORDER BY grade
""").fetchall()

for grade, count in target:
    print(f"  {grade}: {count:,}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)