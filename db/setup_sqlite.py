"""
db/setup_sqlite.py — Create and populate the SQLite databases.

Run this once to create three domain-specific SQLite databases:
    db/airport.db       — employees, flights
    db/tech_startup.db  — employees, projects
    db/restaurant.db    — employees, menus

Each table has ~10 rows of realistic data matching the FAISS schema definitions.
Safe to re-run: drops and recreates all tables each time.
"""

import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# airport
# ─────────────────────────────────────────────────────────────────────────────

def setup_airport(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS flights;

    CREATE TABLE employees (
        employee_id    INTEGER PRIMARY KEY,
        name           TEXT NOT NULL,
        department     TEXT NOT NULL,
        salary         REAL NOT NULL,
        clearance_level INTEGER NOT NULL
    );

    CREATE TABLE flights (
        flight_id   INTEGER PRIMARY KEY,
        airline     TEXT NOT NULL,
        destination TEXT NOT NULL,
        status      TEXT NOT NULL,
        passengers  INTEGER NOT NULL
    );
    """)

    employees = [
        (1,  "James Carter",    "Security",    62000.00, 4),
        (2,  "Maria Lopez",     "Baggage",     45000.00, 2),
        (3,  "David Okafor",    "Gate",        51000.00, 3),
        (4,  "Priya Patel",     "Security",    67000.00, 5),
        (5,  "Tom Nguyen",      "Maintenance", 48000.00, 2),
        (6,  "Sarah Kim",       "Operations",  71000.00, 4),
        (7,  "Robert Singh",    "Baggage",     44000.00, 1),
        (8,  "Linda Brown",     "Gate",        54000.00, 3),
        (9,  "Ahmed Hassan",    "Security",    59000.00, 4),
        (10, "Jessica Martins", "Operations",  68000.00, 3),
    ]
    c.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    flights = [
        (1,  "Delta",     "JFK",  "On Time",  220),
        (2,  "United",    "LAX",  "Delayed",  185),
        (3,  "American",  "ORD",  "On Time",  210),
        (4,  "Southwest", "DEN",  "On Time",  143),
        (5,  "Emirates",  "LHR",  "Boarding", 412),
        (6,  "Delta",     "ATL",  "Delayed",  198),
        (7,  "United",    "SFO",  "On Time",  176),
        (8,  "American",  "DFW",  "Cancelled", 0),
        (9,  "Lufthansa", "FRA",  "On Time",  389),
        (10, "Southwest", "LAS",  "On Time",  160),
    ]
    c.executemany("INSERT INTO flights VALUES (?,?,?,?,?)", flights)

    conn.commit()
    conn.close()
    print(f"  [airport.db] employees=10, flights=10  -> {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# tech_startup
# ─────────────────────────────────────────────────────────────────────────────

def setup_tech_startup(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS projects;

    CREATE TABLE employees (
        employee_id      INTEGER PRIMARY KEY,
        name             TEXT NOT NULL,
        department       TEXT NOT NULL,
        salary           REAL NOT NULL,
        primary_language TEXT NOT NULL
    );

    CREATE TABLE projects (
        project_id INTEGER PRIMARY KEY,
        name       TEXT NOT NULL,
        status     TEXT NOT NULL,
        budget     REAL NOT NULL
    );
    """)

    employees = [
        (1,  "Alex Chen",      "Engineering", 135000.00, "Python"),
        (2,  "Samantha Ray",   "Product",     115000.00, "TypeScript"),
        (3,  "Kevin Park",     "Engineering", 142000.00, "Go"),
        (4,  "Nina Patel",     "Sales",        95000.00, "Python"),
        (5,  "Marcus Green",   "Engineering", 138000.00, "Rust"),
        (6,  "Emily Torres",   "Marketing",    98000.00, "TypeScript"),
        (7,  "Daniel Wu",      "Engineering", 140000.00, "Python"),
        (8,  "Rachel Adams",   "HR",           88000.00, "Java"),
        (9,  "Jason Lee",      "Product",     122000.00, "TypeScript"),
        (10, "Sophie Martin",  "Engineering", 131000.00, "Go"),
    ]
    c.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    projects = [
        (1,  "Project Phoenix",   "Active",    320000.00),
        (2,  "Project Nexus",     "Planning",  150000.00),
        (3,  "Project Atlas",     "Completed", 480000.00),
        (4,  "Project Orion",     "Active",    275000.00),
        (5,  "Project Helix",     "Active",    195000.00),
        (6,  "Project Titan",     "Completed", 520000.00),
        (7,  "Project Aurora",    "Planning",   85000.00),
        (8,  "Project Vortex",    "Active",    340000.00),
        (9,  "Project Meridian",  "Completed", 410000.00),
        (10, "Project Zenith",    "Planning",  120000.00),
    ]
    c.executemany("INSERT INTO projects VALUES (?,?,?,?)", projects)

    conn.commit()
    conn.close()
    print(f"  [tech_startup.db] employees=10, projects=10  -> {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# restaurant
# ─────────────────────────────────────────────────────────────────────────────

def setup_restaurant(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS menus;

    CREATE TABLE employees (
        employee_id INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        department  TEXT NOT NULL,
        salary      REAL NOT NULL,
        shift       TEXT NOT NULL
    );

    CREATE TABLE menus (
        item_id  INTEGER PRIMARY KEY,
        name     TEXT NOT NULL,
        category TEXT NOT NULL,
        price    REAL NOT NULL
    );
    """)

    employees = [
        (1,  "Marco Rossi",    "Kitchen",       55000.00, "Morning"),
        (2,  "Aisha Diallo",   "FrontOfHouse",  38000.00, "Evening"),
        (3,  "Carlos Vega",    "Kitchen",       52000.00, "Night"),
        (4,  "Hannah Scott",   "Management",    72000.00, "Morning"),
        (5,  "Luca Ferrari",   "Kitchen",       49000.00, "Evening"),
        (6,  "Yuki Tanaka",    "Bar",           44000.00, "Night"),
        (7,  "Fatima Ali",     "FrontOfHouse",  36000.00, "Morning"),
        (8,  "Pierre Dupont",  "Kitchen",       58000.00, "Morning"),
        (9,  "Grace Obi",      "Management",    68000.00, "Evening"),
        (10, "Tom Eriksson",   "Bar",           42000.00, "Evening"),
    ]
    c.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    menus = [
        (1,  "Caesar Salad",       "Appetizer", 12.50),
        (2,  "Grilled Salmon",     "Main",      28.00),
        (3,  "Ribeye Steak",       "Main",      42.00),
        (4,  "Margherita Pizza",   "Main",      18.00),
        (5,  "Tiramisu",           "Dessert",    9.00),
        (6,  "Bruschetta",         "Appetizer",  8.50),
        (7,  "Chocolate Fondant",  "Dessert",   10.00),
        (8,  "House Red Wine",     "Drink",     11.00),
        (9,  "Pasta Carbonara",    "Main",      19.50),
        (10, "Lemonade",           "Drink",      5.00),
    ]
    c.executemany("INSERT INTO menus VALUES (?,?,?,?)", menus)

    conn.commit()
    conn.close()
    print(f"  [restaurant.db] employees=10, menus=10  -> {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def setup_all() -> None:
    print("Creating SQLite databases...")
    setup_airport(DB_DIR / "airport.db")
    setup_tech_startup(DB_DIR / "tech_startup.db")
    setup_restaurant(DB_DIR / "restaurant.db")
    print("Done. All databases ready.")


if __name__ == "__main__":
    setup_all()
