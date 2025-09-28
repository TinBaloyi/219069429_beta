from config import DB_CONFIG
import psycopg2

def get_connection():
    """Create and return a new database connection using DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)

def my_db_function():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users;")
    rows = cur.fetchall()
    for row in rows:
        print(row)
    cur.close()
    conn.close()

def add_user(username, email, password_hash, first_name, last_name, role_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (username, email, password_hash, first_name, last_name, role_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (username, email, password_hash, first_name, last_name, role_id))
    conn.commit()
    cur.close()
    conn.close()
