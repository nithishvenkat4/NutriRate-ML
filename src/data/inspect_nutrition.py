import duckdb

FILE = r"D:\NutriRateAI\food.parquet"

db = duckdb.connect()

query = f"""
SELECT
    code,
    product_name,
    nutriscore_grade,
    nutriscore_score,
    nutriments
FROM read_parquet('{FILE}')
WHERE nutriscore_grade IS NOT NULL
LIMIT 10
"""

rows = db.execute(query).fetchall()

for row in rows:
    print("=" * 100)
    print("CODE:", row[0])
    print("PRODUCT:", row[1])
    print("NUTRI-SCORE GRADE:", row[2])
    print("NUTRI-SCORE SCORE:", row[3])
    print("NUTRIMENTS:")
    
    if row[4]:
        for nutrient in row[4]:
            print(nutrient)