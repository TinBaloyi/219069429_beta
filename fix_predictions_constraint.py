import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "bottleneck",
    "user": "postgres",
    "password": "Tintswalo@2"  # UPDATE THIS
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("=" * 60)
print("Checking predictions constraints:")
print("=" * 60)

# Get all constraints on predictions table
cur.execute("""
    SELECT conname, pg_get_constraintdef(oid)
    FROM pg_constraint 
    WHERE conrelid = 'predictions'::regclass 
    AND contype = 'c'
""")

results = cur.fetchall()
if results:
    for name, definition in results:
        print(f"\nConstraint: {name}")
        print(f"Definition: {definition}")
        
        # Extract allowed values if it's a CHECK constraint
        import re
        matches = re.findall(r"'([^']*)'", definition)
        if matches:
            print("Allowed values:")
            for val in matches:
                print(f"  - {val}")
else:
    print("No CHECK constraints found")

# Also check if there are any existing values in the table
print("\n" + "=" * 60)
print("Checking existing prediction_type values:")
print("=" * 60)
cur.execute("SELECT DISTINCT prediction_type FROM predictions")
existing = cur.fetchall()
if existing:
    for val in existing:
        print(f"  - {val[0]}")
else:
    print("  (No data in table yet)")

cur.close()
conn.close()