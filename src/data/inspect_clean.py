import duckdb

FILE = r"D:\NutriRateAI\nutrirate_ml_clean.parquet"

db = duckdb.connect()

print("=" * 75)
print("NUTRIRATE AI - CLEAN DATASET INSPECTION")
print("=" * 75)

# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

info = db.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT code) AS unique_codes
    FROM read_parquet('{FILE}')
""").fetchone()

print(f"\nRows:          {info[0]:,}")
print(f"Unique codes:  {info[1]:,}")


# ---------------------------------------------------------
# COLUMNS
# ---------------------------------------------------------

columns = db.execute(f"""
    DESCRIBE SELECT *
    FROM read_parquet('{FILE}')
""").fetchall()

print(f"\nColumns: {len(columns)}")

for col in columns:
    print(f"  {col[0]:30s} {col[1]}")


# ---------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------

print("\nSample records:")

rows = db.execute(f"""
    SELECT *
    FROM read_parquet('{FILE}')
    LIMIT 5
""").fetchall()

column_names = [c[0] for c in columns]

for i, row in enumerate(rows, 1):

    print("\n" + "-" * 75)
    print(f"Record {i}")

    for name, value in zip(column_names, row):
        print(f"{name}: {value}")


print("\n" + "=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)