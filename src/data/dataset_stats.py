import duckdb

FILE = r"D:\NutriRateAI\food.parquet"

db = duckdb.connect()

print("=" * 70)
print("NUTRIRATE AI - DATASET QUALITY ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. TOTAL PRODUCTS
# ---------------------------------------------------------

total = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
""").fetchone()[0]

print(f"\nTotal products: {total:,}")


# ---------------------------------------------------------
# 2. NUTRI-SCORE DISTRIBUTION
# ---------------------------------------------------------

print("\nNutri-Score distribution:")

grades = db.execute(f"""
    SELECT
        LOWER(TRIM(nutriscore_grade)) AS grade,
        COUNT(*) AS count
    FROM read_parquet('{FILE}')
    WHERE LOWER(TRIM(nutriscore_grade))
          IN ('a', 'b', 'c', 'd', 'e')
    GROUP BY grade
    ORDER BY grade
""").fetchall()

for grade, count in grades:
    percentage = count / total * 100
    print(f"  {grade.upper()}: {count:,} ({percentage:.2f}%)")


# ---------------------------------------------------------
# 3. NUTRI-SCORE SCORE
# ---------------------------------------------------------

with_score = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
    WHERE nutriscore_score IS NOT NULL
""").fetchone()[0]

print(f"\nProducts with Nutri-Score score: {with_score:,}")


# ---------------------------------------------------------
# 4. NUTRITION DATA
# ---------------------------------------------------------

nutrition = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
    WHERE nutriments IS NOT NULL
      AND no_nutrition_data = FALSE
""").fetchone()[0]

print(f"Products with nutrition data: {nutrition:,}")


# ---------------------------------------------------------
# 5. INGREDIENTS
# ---------------------------------------------------------

ingredients = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
    WHERE ingredients IS NOT NULL
      AND LENGTH(TRIM(ingredients)) > 0
""").fetchone()[0]

print(f"Products with ingredients: {ingredients:,}")


# ---------------------------------------------------------
# 6. IMAGES
# ---------------------------------------------------------

images = db.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}')
    WHERE images IS NOT NULL
      AND ARRAY_LENGTH(images) > 0
""").fetchone()[0]

print(f"Products with images: {images:,}")


# ---------------------------------------------------------
# 7. NUTRIENT AVAILABILITY
# ---------------------------------------------------------

required_nutrients = [
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "proteins",
    "fiber",
    "salt"
]

print("\nChecking core nutrient availability...")

for nutrient in required_nutrients:

    query = f"""
        SELECT COUNT(*)
        FROM read_parquet('{FILE}') AS p
        WHERE EXISTS (
            SELECT 1
            FROM UNNEST(p.nutriments) AS u(n)
            WHERE n.name = '{nutrient}'
              AND n."100g" IS NOT NULL
        )
    """

    count = db.execute(query).fetchone()[0]

    percentage = count / total * 100

    print(
        f"  {nutrient:20s}: "
        f"{count:,} ({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# 8. COMPLETE CORE NUTRITION
# ---------------------------------------------------------

print("\nChecking products with all 8 core nutrients...")

conditions = []

for nutrient in required_nutrients:

    conditions.append(f"""
        EXISTS (
            SELECT 1
            FROM UNNEST(p.nutriments) AS u(n)
            WHERE n.name = '{nutrient}'
              AND n."100g" IS NOT NULL
        )
    """)

all_conditions = "\nAND ".join(conditions)

complete_query = f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}') AS p
    WHERE {all_conditions}
"""

complete = db.execute(complete_query).fetchone()[0]

print(
    f"Products with ALL 8 core nutrients: "
    f"{complete:,}"
)


# ---------------------------------------------------------
# 9. POTENTIAL ANN TRAINING DATA
# ---------------------------------------------------------

print("\nFinding potential ANN training products...")

training_query = f"""
    SELECT COUNT(*)
    FROM read_parquet('{FILE}') AS p
    WHERE nutriscore_score IS NOT NULL
      AND LOWER(TRIM(nutriscore_grade))
          IN ('a', 'b', 'c', 'd', 'e')
      AND {all_conditions}
"""

training = db.execute(training_query).fetchone()[0]

print(
    f"Potential ANN training products: "
    f"{training:,}"
)


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)