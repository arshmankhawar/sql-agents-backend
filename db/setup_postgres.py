"""
db/setup_postgres.py — Create and populate the unified PostgreSQL database.

Run this once (against a fresh Postgres instance, e.g. via `docker compose up`)
to create every domain's data in shared base tables discriminated by a
`domain` column:

    employees (domain, employee_id, name, department, salary, + domain extras)
    flights   (domain, ...)   -- airport
    projects  (domain, ...)   -- tech_startup
    menus     (domain, ...)   -- restaurant

Domain isolation is enforced at the DB layer through per-domain VIEWS. Each
view exposes only one domain's rows and only the columns relevant to that
domain, e.g.:

    airport_employees      -> employees WHERE domain='airport'      (+ clearance_level)
    tech_startup_employees -> employees WHERE domain='tech_startup' (+ primary_language)
    restaurant_employees   -> employees WHERE domain='restaurant'   (+ shift)

Agents query the VIEWS, never the base tables. This means:
  - An agent scoped to one domain physically cannot read another domain's rows.
  - SQL agents never need (and must never write) a `WHERE domain = '...'` clause;
    the view already encapsulates that filter.

Replaces the old db/setup_sqlite.py (single-file SQLite -> unified Postgres
database). Safe to re-run: drops and recreates all tables and views each time.
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Make the project root importable so this script can be run directly
# (python db/setup_postgres.py) and still import utils.passwords / config.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_URL  # noqa: E402
from utils.passwords import hash_password  # noqa: E402

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Schema — shared base tables (domain-discriminated)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
DROP VIEW  IF EXISTS airport_employees;
DROP VIEW  IF EXISTS airport_flights;
DROP VIEW  IF EXISTS tech_startup_employees;
DROP VIEW  IF EXISTS tech_startup_projects;
DROP VIEW  IF EXISTS restaurant_employees;
DROP VIEW  IF EXISTS restaurant_menus;

DROP TABLE IF EXISTS document_chunks;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS uploaded_datasets;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS flights;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS menus;

-- One employees table for every domain. employee_id is no longer globally
-- unique (each domain numbers from 1), so a surrogate `id` is the primary key
-- and (domain, employee_id) is unique. Domain-specific attributes are nullable.
CREATE TABLE employees (
    id               SERIAL PRIMARY KEY,
    domain           TEXT    NOT NULL,
    employee_id      INTEGER NOT NULL,
    name             TEXT    NOT NULL,
    department       TEXT    NOT NULL,
    salary           REAL    NOT NULL,
    clearance_level  INTEGER,          -- airport only
    primary_language TEXT,             -- tech_startup only
    shift            TEXT,             -- restaurant only
    UNIQUE (domain, employee_id)
);

CREATE TABLE flights (
    id          SERIAL PRIMARY KEY,
    domain      TEXT    NOT NULL,
    flight_id   INTEGER NOT NULL,
    airline     TEXT    NOT NULL,
    destination TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    passengers  INTEGER NOT NULL,
    UNIQUE (domain, flight_id)
);

CREATE TABLE projects (
    id         SERIAL PRIMARY KEY,
    domain     TEXT    NOT NULL,
    project_id INTEGER NOT NULL,
    name       TEXT    NOT NULL,
    status     TEXT    NOT NULL,
    budget     REAL    NOT NULL,
    UNIQUE (domain, project_id)
);

CREATE TABLE menus (
    id       SERIAL PRIMARY KEY,
    domain   TEXT    NOT NULL,
    item_id  INTEGER NOT NULL,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    REAL    NOT NULL,
    UNIQUE (domain, item_id)
);

-- Application users for JWT login. Passwords are bcrypt-hashed (never stored
-- in plaintext). Seeded with an admin user from ADMIN_USERNAME/ADMIN_PASSWORD.
CREATE TABLE users (
    id            SERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Uploaded data registry (CSV/Excel → DB) ──────────────────────────────────
-- One row per uploaded tabular dataset. The actual data lives in a dynamically
-- created `user_<name>` table (+ a same-named view) so the SQL pipeline queries
-- it exactly like the built-in domain views.
CREATE TABLE uploaded_datasets (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,   -- logical dataset name (also the view name)
    table_name  TEXT NOT NULL,          -- physical base table name (user_<name>)
    domain      TEXT NOT NULL,          -- always the uploads domain
    columns     TEXT NOT NULL,          -- JSON: [{"name","type"}, ...]
    row_count   INTEGER NOT NULL DEFAULT 0,
    filename    TEXT,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Uploaded documents (file search / vector store) ──────────────────────────
-- Metadata for each uploaded document. Chunk text + embeddings are tracked in
-- document_chunks; the embeddings themselves live in a FAISS index on disk.
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    description TEXT,
    file_type   TEXT,                   -- pdf | txt | docx | md
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,       -- ordinal position within the document
    text        TEXT NOT NULL,
    heading     TEXT,                  -- section heading this chunk falls under, if any
    category    TEXT NOT NULL DEFAULT 'general'  -- rule-based tag: policy/financial/narrative/table/general
);

-- ── Per-domain views (the surface agents query) ──────────────────────────────
-- Each view filters to one domain and projects only that domain's columns, so
-- the schema an agent sees is identical to the old per-file schema.

CREATE VIEW airport_employees AS
    SELECT employee_id, name, department, salary, clearance_level
    FROM employees WHERE domain = 'airport';

CREATE VIEW airport_flights AS
    SELECT flight_id, airline, destination, status, passengers
    FROM flights WHERE domain = 'airport';

CREATE VIEW tech_startup_employees AS
    SELECT employee_id, name, department, salary, primary_language
    FROM employees WHERE domain = 'tech_startup';

CREATE VIEW tech_startup_projects AS
    SELECT project_id, name, status, budget
    FROM projects WHERE domain = 'tech_startup';

CREATE VIEW restaurant_employees AS
    SELECT employee_id, name, department, salary, shift
    FROM employees WHERE domain = 'restaurant';

CREATE VIEW restaurant_menus AS
    SELECT item_id, name, category, price
    FROM menus WHERE domain = 'restaurant';
"""


# ─────────────────────────────────────────────────────────────────────────────
# Seed data
# ─────────────────────────────────────────────────────────────────────────────

# employees: (domain, employee_id, name, department, salary, clearance_level, primary_language, shift)
_EMPLOYEES = [
    # airport — clearance_level set, others NULL
    ("airport", 1,  "James Carter",    "Security",    62000.00, 4, None, None),
    ("airport", 2,  "Maria Lopez",     "Baggage",     45000.00, 2, None, None),
    ("airport", 3,  "David Okafor",    "Gate",        51000.00, 3, None, None),
    ("airport", 4,  "Priya Patel",     "Security",    67000.00, 5, None, None),
    ("airport", 5,  "Tom Nguyen",      "Maintenance", 48000.00, 2, None, None),
    ("airport", 6,  "Sarah Kim",       "Operations",  71000.00, 4, None, None),
    ("airport", 7,  "Robert Singh",    "Baggage",     44000.00, 1, None, None),
    ("airport", 8,  "Linda Brown",     "Gate",        54000.00, 3, None, None),
    ("airport", 9,  "Ahmed Hassan",    "Security",    59000.00, 4, None, None),
    ("airport", 10, "Jessica Martins", "Operations",  68000.00, 3, None, None),
    # tech_startup — primary_language set
    ("tech_startup", 1,  "Alex Chen",     "Engineering", 135000.00, None, "Python",     None),
    ("tech_startup", 2,  "Samantha Ray",  "Product",     115000.00, None, "TypeScript", None),
    ("tech_startup", 3,  "Kevin Park",    "Engineering", 142000.00, None, "Go",         None),
    ("tech_startup", 4,  "Nina Patel",    "Sales",        95000.00, None, "Python",     None),
    ("tech_startup", 5,  "Marcus Green",  "Engineering", 138000.00, None, "Rust",       None),
    ("tech_startup", 6,  "Emily Torres",  "Marketing",    98000.00, None, "TypeScript", None),
    ("tech_startup", 7,  "Daniel Wu",     "Engineering", 140000.00, None, "Python",     None),
    ("tech_startup", 8,  "Rachel Adams",  "HR",           88000.00, None, "Java",       None),
    ("tech_startup", 9,  "Jason Lee",     "Product",     122000.00, None, "TypeScript", None),
    ("tech_startup", 10, "Sophie Martin", "Engineering", 131000.00, None, "Go",         None),
    # restaurant — shift set
    ("restaurant", 1,  "Marco Rossi",   "Kitchen",      55000.00, None, None, "Morning"),
    ("restaurant", 2,  "Aisha Diallo",  "FrontOfHouse", 38000.00, None, None, "Evening"),
    ("restaurant", 3,  "Carlos Vega",   "Kitchen",      52000.00, None, None, "Night"),
    ("restaurant", 4,  "Hannah Scott",  "Management",   72000.00, None, None, "Morning"),
    ("restaurant", 5,  "Luca Ferrari",  "Kitchen",      49000.00, None, None, "Evening"),
    ("restaurant", 6,  "Yuki Tanaka",   "Bar",          44000.00, None, None, "Night"),
    ("restaurant", 7,  "Fatima Ali",    "FrontOfHouse", 36000.00, None, None, "Morning"),
    ("restaurant", 8,  "Pierre Dupont", "Kitchen",      58000.00, None, None, "Morning"),
    ("restaurant", 9,  "Grace Obi",     "Management",   68000.00, None, None, "Evening"),
    ("restaurant", 10, "Tom Eriksson",  "Bar",          42000.00, None, None, "Evening"),
]

# flights: (domain, flight_id, airline, destination, status, passengers)
_FLIGHTS = [
    ("airport", 1,  "Delta",     "JFK", "On Time",   220),
    ("airport", 2,  "United",    "LAX", "Delayed",   185),
    ("airport", 3,  "American",  "ORD", "On Time",   210),
    ("airport", 4,  "Southwest", "DEN", "On Time",   143),
    ("airport", 5,  "Emirates",  "LHR", "Boarding",  412),
    ("airport", 6,  "Delta",     "ATL", "Delayed",   198),
    ("airport", 7,  "United",    "SFO", "On Time",   176),
    ("airport", 8,  "American",  "DFW", "Cancelled",   0),
    ("airport", 9,  "Lufthansa", "FRA", "On Time",   389),
    ("airport", 10, "Southwest", "LAS", "On Time",   160),
]

# projects: (domain, project_id, name, status, budget)
_PROJECTS = [
    ("tech_startup", 1,  "Project Phoenix",  "Active",    320000.00),
    ("tech_startup", 2,  "Project Nexus",    "Planning",  150000.00),
    ("tech_startup", 3,  "Project Atlas",    "Completed", 480000.00),
    ("tech_startup", 4,  "Project Orion",    "Active",    275000.00),
    ("tech_startup", 5,  "Project Helix",    "Active",    195000.00),
    ("tech_startup", 6,  "Project Titan",    "Completed", 520000.00),
    ("tech_startup", 7,  "Project Aurora",   "Planning",   85000.00),
    ("tech_startup", 8,  "Project Vortex",   "Active",    340000.00),
    ("tech_startup", 9,  "Project Meridian", "Completed", 410000.00),
    ("tech_startup", 10, "Project Zenith",   "Planning",  120000.00),
]

# menus: (domain, item_id, name, category, price)
_MENUS = [
    ("restaurant", 1,  "Caesar Salad",      "Appetizer", 12.50),
    ("restaurant", 2,  "Grilled Salmon",    "Main",      28.00),
    ("restaurant", 3,  "Ribeye Steak",      "Main",      42.00),
    ("restaurant", 4,  "Margherita Pizza",  "Main",      18.00),
    ("restaurant", 5,  "Tiramisu",          "Dessert",    9.00),
    ("restaurant", 6,  "Bruschetta",        "Appetizer",  8.50),
    ("restaurant", 7,  "Chocolate Fondant", "Dessert",   10.00),
    ("restaurant", 8,  "House Red Wine",    "Drink",     11.00),
    ("restaurant", 9,  "Pasta Carbonara",   "Main",      19.50),
    ("restaurant", 10, "Lemonade",          "Drink",      5.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def setup_all() -> None:
    print(f"Creating unified PostgreSQL database (DSN={DATABASE_URL!r})...")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as c:
            c.execute(_SCHEMA)

            c.executemany(
                "INSERT INTO employees "
                "(domain, employee_id, name, department, salary, clearance_level, primary_language, shift) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                _EMPLOYEES,
            )
            c.executemany(
                "INSERT INTO flights (domain, flight_id, airline, destination, status, passengers) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                _FLIGHTS,
            )
            c.executemany(
                "INSERT INTO projects (domain, project_id, name, status, budget) VALUES (%s,%s,%s,%s,%s)",
                _PROJECTS,
            )
            c.executemany(
                "INSERT INTO menus (domain, item_id, name, category, price) VALUES (%s,%s,%s,%s,%s)",
                _MENUS,
            )

            # Seed the admin user (bcrypt-hashed). Credentials come from the
            # environment so the password is never committed to source.
            admin_user = os.getenv("ADMIN_USERNAME", "admin")
            admin_pass = os.getenv("ADMIN_PASSWORD", "change-me")
            c.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (admin_user, hash_password(admin_pass)),
            )

        conn.commit()
    finally:
        conn.close()

    print(f"  employees={len(_EMPLOYEES)}, flights={len(_FLIGHTS)}, "
          f"projects={len(_PROJECTS)}, menus={len(_MENUS)}")
    print("  views: airport_employees, airport_flights, tech_startup_employees, "
          "tech_startup_projects, restaurant_employees, restaurant_menus")
    print(f"  users: 1 (admin='{os.getenv('ADMIN_USERNAME', 'admin')}')")
    print("Done. Unified Postgres database ready.")


if __name__ == "__main__":
    setup_all()
