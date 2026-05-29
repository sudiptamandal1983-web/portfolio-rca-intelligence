"""
data_connector.py — Unified data source connector.

Loads tabular data from CSV, DuckDB, or Postgres into a DuckDB
in-memory connection that the rest of the pipeline queries against.

Why DuckDB as the unified layer:
    - SQL interface works identically regardless of source
    - Zero-copy reads from CSV and Pandas DataFrames
    - No server needed for CSV or file-based sources
    - HunterAgent's SQL queries run unchanged

Usage:
    from agents.data_connector import DataConnector

    # From CSV
    con = DataConnector.from_csv("data/sales.csv", table_name="loans")

    # From existing DuckDB file
    con = DataConnector.from_duckdb("data/portfolio.db")

    # From Postgres
    con = DataConnector.from_postgres(
        "postgresql://user:password@host:5432/dbname",
        table="loans"
    )

    # From config dict
    con = DataConnector.from_config(config["data"])
"""

import os
import duckdb
import pandas as pd
from typing import Optional


class DataConnector:
    """
    Wraps a DuckDB connection and exposes a unified query interface
    regardless of the underlying data source.

    The connection always has a table called `loans` (or the configured
    table_name) that HunterAgent queries against. This means HunterAgent's
    SQL works unchanged on any data source.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, table_name: str = "loans"):
        self._con        = con
        self.table_name  = table_name
        self._source     = "unknown"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path:       str,
        table_name: str = "loans",
        sample_rows: Optional[int] = None,
    ) -> "DataConnector":
        """
        Loads a CSV file into an in-memory DuckDB table.

        Parameters
        ----------
        path        : Path to CSV file
        table_name  : Name to register the table as (default: loans)
        sample_rows : If set, only loads this many rows (for testing)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        con = duckdb.connect(":memory:")

        if sample_rows:
            con.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{path}')
                LIMIT {sample_rows}
            """)
        else:
            con.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{path}')
            """)

        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  📂  Loaded CSV: {path} → {row_count:,} rows → table '{table_name}'")

        instance          = cls(con, table_name)
        instance._source  = f"csv:{path}"
        return instance

    @classmethod
    def from_duckdb(
        cls,
        db_path:    str,
        table_name: str = "loans",
    ) -> "DataConnector":
        """
        Connects to an existing DuckDB file.

        Parameters
        ----------
        db_path    : Path to .db file
        table_name : Table to use (default: loans)
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DuckDB file not found: {db_path}")

        con = duckdb.connect(db_path)

        # Verify table exists
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        if table_name not in tables:
            raise ValueError(
                f"Table '{table_name}' not found in {db_path}. "
                f"Available tables: {tables}"
            )

        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  🗄️   Connected to DuckDB: {db_path} → {row_count:,} rows → table '{table_name}'")

        instance         = cls(con, table_name)
        instance._source = f"duckdb:{db_path}"
        return instance

    @classmethod
    def from_postgres(
        cls,
        connection_string: str,
        table:             str,
        schema:            str = "public",
        table_name:        str = "loans",
    ) -> "DataConnector":
        """
        Reads a Postgres table into an in-memory DuckDB table.

        Parameters
        ----------
        connection_string : PostgreSQL connection string
                            e.g. postgresql://user:pass@host:5432/dbname
        table             : Source table name in Postgres
        schema            : Postgres schema (default: public)
        table_name        : Name to register as in DuckDB (default: loans)
        """
        try:
            import psycopg2
            import sqlalchemy
        except ImportError:
            raise ImportError(
                "Postgres support requires psycopg2 and sqlalchemy.\n"
                "Run: pip install psycopg2-binary sqlalchemy"
            )

        print(f"  🐘  Connecting to Postgres: {connection_string[:40]}...")

        # Read via pandas then load into DuckDB
        engine = sqlalchemy.create_engine(connection_string)
        df     = pd.read_sql_table(table, engine, schema=schema)

        con = duckdb.connect(":memory:")
        con.register(table_name, df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}")

        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  ✅  Loaded Postgres table '{schema}.{table}' → {row_count:,} rows")

        instance         = cls(con, table_name)
        instance._source = f"postgres:{table}"
        return instance

    @classmethod
    def from_config(cls, data_config: dict) -> "DataConnector":
        """
        Creates a DataConnector from a config dict (from config.yaml).

        Expected keys:
            source     : csv | duckdb | postgres
            path       : file path (csv/duckdb) or connection string (postgres)
            table      : table name (duckdb/postgres, default: loans)
            table_name : what to register it as in DuckDB (default: loans)
            sample_rows: optional row limit for testing
        """
        source     = data_config.get("source", "duckdb").lower()
        table_name = data_config.get("table_name", data_config.get("table", "loans"))

        if source == "csv":
            return cls.from_csv(
                path        = data_config["path"],
                table_name  = table_name,
                sample_rows = data_config.get("sample_rows"),
            )
        elif source == "duckdb":
            return cls.from_duckdb(
                db_path    = data_config["path"],
                table_name = table_name,
            )
        elif source == "postgres":
            return cls.from_postgres(
                connection_string = data_config["path"],
                table             = data_config.get("table", "loans"),
                schema            = data_config.get("schema", "public"),
                table_name        = table_name,
            )
        else:
            raise ValueError(
                f"Unknown data source '{source}'. "
                f"Choose from: csv | duckdb | postgres"
            )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def execute(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns a DataFrame."""
        return self._con.execute(query).df()

    def get_columns(self) -> list[str]:
        """Returns column names for the main table."""
        return self._con.execute(
            f"PRAGMA table_info({self.table_name})"
        ).df()["name"].tolist()

    def get_column_map(self) -> dict:
        """
        Returns normalised key → actual column name mapping.
        Normalised = lowercase with underscores removed.
        Used by HunterAgent for defensive column lookup.
        """
        cols = self.get_columns()
        return {c.lower().replace("_", ""): c for c in cols}

    def row_count(self) -> int:
        """Returns total row count."""
        return self._con.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()[0]

    def sample(self, n: int = 5) -> pd.DataFrame:
        """Returns first N rows for inspection."""
        return self._con.execute(
            f"SELECT * FROM {self.table_name} LIMIT {n}"
        ).df()

    def close(self):
        """Closes the DuckDB connection."""
        if self._con:
            self._con.close()
            self._con = None

    def __repr__(self):
        return f"DataConnector(source={self._source}, table={self.table_name})"
