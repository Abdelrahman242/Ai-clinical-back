from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}
engine_options = {
    "connect_args": connect_args,
    "pool_pre_ping": True,
}

# Supabase Session Pooler is PostgreSQL. Keep the pool deliberately small for
# serverless/container deployments and recycle connections periodically.
if not is_sqlite:
    engine_options.update({"pool_size": 5, "max_overflow": 10, "pool_recycle": 1800})

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
