"""Add bearing_type, coating_status, size_info, source_notes to ImageRecord if missing."""
import sqlite3
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, 'image_database.db')
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(ROOT, 'instance', 'image_database.db')

COLUMNS = [
    ('bearing_type', 'VARCHAR(50)'),
    ('coating_status', 'VARCHAR(20)'),
    ('size_info', 'VARCHAR(100)'),
    ('source_notes', 'TEXT'),
    ('split', 'VARCHAR(20)'),
]

def main():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}; init_db will create it.")
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(image_record)")
    existing = {row[1] for row in cur.fetchall()}
    for col, typ in COLUMNS:
        if col not in existing:
            cur.execute(f"ALTER TABLE image_record ADD COLUMN {col} {typ}")
            print(f"Added column {col}")
    conn.commit()
    conn.close()
    print("Migration done.")

if __name__ == "__main__":
    main()
