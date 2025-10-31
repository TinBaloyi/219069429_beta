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
print("Checking pipeline_metrics constraints:")
print("=" * 60)

# Get the constraint definition
cur.execute("""
    SELECT pg_get_constraintdef(oid)
    FROM pg_constraint 
    WHERE conrelid = 'pipeline_metrics'::regclass 
    AND conname LIKE '%stage%'
""")

result = cur.fetchone()
if result:
    print("\nConstraint definition:")
    print(result[0])
    
    # Extract allowed values
    import re
    matches = re.findall(r"'([^']*)'", result[0])
    print("\nAllowed values:")
    for val in matches:
        print(f"  - {val}")
else:
    print("No constraint found")

cur.close()
conn.close()