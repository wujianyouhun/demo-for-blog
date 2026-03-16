from sqlalchemy import create_engine
import os

DB_URL = os.getenv(
"DB_URL",
"postgresql+psycopg2://postgres:postgres@localhost:5432/spatial"
)

engine = create_engine(DB_URL)

def run_sql(sql):

    with engine.connect() as conn:

        result = conn.execute(sql)

        return [dict(row) for row in result]
