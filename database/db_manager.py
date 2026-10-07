"""
database/db_manager.py
Handles SQLite database connection, table initialization, seeding, and querying.
Designed cleanly for beginners and easy viva explanation.
"""

import os
import sqlite3
import pandas as pd
from data.seed_data import LAPTOP_DATA

# Path to the SQLite database file
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "laptops.db")


def get_connection(read_only: bool = False) -> sqlite3.Connection:
    """
    Establish and return a connection to the SQLite database.
    If read_only is True, establishes a URI connection in read-only mode ('mode=ro'),
    preventing any INSERT, UPDATE, DELETE, or DDL operations at the SQLite engine level.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    if read_only and os.path.exists(DB_PATH):
        abs_path = os.path.abspath(DB_PATH).replace("\\", "/")
        uri = f"file:{abs_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Initializes the database schema and seeds realistic laptop data
    if the table is empty or does not exist. Uses write connection.
    """
    conn = get_connection(read_only=False)
    cursor = conn.cursor()

    # Create laptops table with complete specifications and domain scores
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laptops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            price REAL NOT NULL,
            processor TEXT NOT NULL,
            ram INTEGER NOT NULL,
            storage TEXT NOT NULL,
            gpu TEXT NOT NULL,
            gpu_type TEXT NOT NULL,
            display TEXT NOT NULL,
            battery TEXT NOT NULL,
            weight REAL NOT NULL,
            gaming_score REAL NOT NULL,
            programming_score REAL NOT NULL,
            ai_ml_score REAL NOT NULL,
            video_editing_score REAL NOT NULL
        )
    """)
    conn.commit()

    # Check if table has data or needs upgrade to expanded dataset
    cursor.execute("SELECT COUNT(*) FROM laptops")
    count = cursor.fetchone()[0]

    if count < len(LAPTOP_DATA):
        cursor.execute("DELETE FROM laptops")
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='laptops'")
        except Exception:
            pass
        insert_query = """
            INSERT INTO laptops (
                brand, model, price, processor, ram, storage, gpu,
                gpu_type, display, battery, weight, gaming_score,
                programming_score, ai_ml_score, video_editing_score
            ) VALUES (
                :brand, :model, :price, :processor, :ram, :storage, :gpu,
                :gpu_type, :display, :battery, :weight, :gaming_score,
                :programming_score, :ai_ml_score, :video_editing_score
            )
        """
        cursor.executemany(insert_query, LAPTOP_DATA)
        conn.commit()

    conn.close()


def get_all_laptops() -> pd.DataFrame:
    """Fetch all laptops from the database into a Pandas DataFrame using a read-only connection."""
    conn = get_connection(read_only=True)
    df = pd.read_sql_query("SELECT * FROM laptops ORDER BY price ASC", conn)
    conn.close()
    return df


def get_unique_brands() -> list:
    """Return sorted list of distinct laptop brands using a read-only connection."""
    conn = get_connection(read_only=True)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT brand FROM laptops ORDER BY brand ASC")
    brands = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brands


def get_budget_bounds() -> tuple:
    """Return min and max laptop price in the database using a read-only connection."""
    conn = get_connection(read_only=True)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(price), MAX(price) FROM laptops")
    min_price, max_price = cursor.fetchone()
    conn.close()
    return int(min_price or 30000), int(max_price or 350000)
