"""
Load Parquet data from data/Sample into SQL Server.

Each subfolder under data/Sample is named like `dbo_<table>` and contains one or
more Spark-style parquet files (sometimes nested inside a GUID subfolder).
The folder name maps to a SQL Server schema and table:
    dbo_accident        -> schema "dbo", table "accident"
    dbo_claims_extra    -> schema "dbo", table "claims_extra"

Usage:
    python load_parquet_to_sqlserver.py
    python load_parquet_to_sqlserver.py --database MyDb --if-exists append
    python load_parquet_to_sqlserver.py --tables dbo_accident dbo_address
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine

# --- Defaults -------------------------------------------------------------
SERVER = r"RADIANT-SYSTEM\radiantsystem"
DATABASE = "Learning"          # change to your target database
# Workspace root is one level up from this script's folder (script/).
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Sample"
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
CHUNK_SIZE = 10_000            # rows per insert batch


def build_engine(server: str, database: str, driver: str):
    """Create a SQLAlchemy engine using Windows (trusted) authentication."""
    params = urllib.parse.quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
    )


def split_schema_table(folder_name: str) -> tuple[str, str]:
    """`dbo_claims_extra` -> ("dbo", "claims_extra")."""
    schema, _, table = folder_name.partition("_")
    if not table:
        # No underscore: treat whole name as table in default schema.
        return "dbo", folder_name
    return schema, table


def normalize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime columns to tz-naive (UTC wall time).

    SQLAlchemy maps tz-aware datetimes to SQL Server's TIMESTAMP type, which is a
    non-insertable rowversion. Stripping the timezone yields a normal DATETIME.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def read_folder_parquet(folder: Path) -> pd.DataFrame | None:
    """Read the canonical parquet file(s) for a table folder.

    Spark writes the table as top-level `part-*.parquet` files alongside a
    `_SUCCESS` marker. Some folders also contain a nested copy in a GUID
    subfolder; we ignore those to avoid duplicate/conflicting schemas. Only if no
    top-level parquet exists do we fall back to a recursive search.
    """
    parquet_files = sorted(folder.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(folder.rglob("*.parquet"))
    if not parquet_files:
        return None
    frames = [pq.read_table(p).to_pandas() for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    return normalize_datetimes(df)


def load_folder(engine, folder: Path, if_exists: str) -> int:
    schema, table = split_schema_table(folder.name)
    df = read_folder_parquet(folder)
    if df is None:
        print(f"  [skip] no parquet files in {folder.name}", flush=True)
        return 0

    # Use simple parameterized inserts (no fast_executemany). The dataset is
    # small, and fast_executemany triggers an "HY090 invalid string/buffer
    # length" pyodbc bug on VARCHAR(max) columns containing empty strings.
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=CHUNK_SIZE,
    )
    print(f"  [ok]   {schema}.{table}: {len(df):,} rows", flush=True)
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load Sample parquet data into SQL Server.")
    parser.add_argument("--server", default=SERVER, help="SQL Server instance.")
    parser.add_argument("--database", default=DATABASE, help="Target database name.")
    parser.add_argument("--driver", default=ODBC_DRIVER, help="ODBC driver name.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to data/Sample.")
    parser.add_argument(
        "--if-exists",
        default="replace",
        choices=["fail", "replace", "append"],
        help="Behavior when the target table already exists.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        help="Optional list of folder names to load (default: all dbo_* folders).",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    folders = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if args.tables:
        wanted = set(args.tables)
        folders = [p for p in folders if p.name in wanted]
        missing = wanted - {p.name for p in folders}
        for name in sorted(missing):
            print(f"  [warn] requested folder not found: {name}", file=sys.stderr)

    if not folders:
        print("No folders to load.", file=sys.stderr)
        return 1

    print(f"Connecting to {args.server} / {args.database} ...")
    engine = build_engine(args.server, args.database, args.driver)

    # Fail fast if the connection is bad.
    with engine.connect():
        pass

    total_rows = 0
    loaded = 0
    failed = 0
    print(f"Loading {len(folders)} folder(s) from {data_dir} (if_exists={args.if_exists})")
    for folder in folders:
        try:
            rows = load_folder(engine, folder, args.if_exists)
            total_rows += rows
            if rows:
                loaded += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  [fail] {folder.name}: {exc}", file=sys.stderr)

    print(f"\nDone. Tables loaded: {loaded}, failed: {failed}, total rows: {total_rows:,}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    main()
