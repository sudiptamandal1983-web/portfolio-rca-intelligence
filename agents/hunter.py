import duckdb
import pandas as pd
from typing import Optional, Union


def build_dimension_registry(config_dimensions: list) -> dict:
    """
    Builds a DIMENSION_REGISTRY-compatible dict from config.yaml dimensions list.
    Allows the pipeline to work with any dataset defined in config.
    """
    registry = {}
    for dim in config_dimensions:
        name = dim["name"]
        registry[name] = {
            "description": dim.get("description", name),
            "dimensions":  dim["columns"],
            "metric_sql":  dim["metric_sql"].strip(),
            "metric_name": dim["metric_name"],
            "insight_cols": {
                "key":          dim["columns"][0],
                "metric_label": dim["metric_name"],
            },
        }
    return registry




# ---------------------------------------------------------------------------
# Dimension registry — single source of truth for all audit dimensions.
# Add a new dimension here; the rest of the pipeline picks it up automatically.
# ---------------------------------------------------------------------------
DIMENSION_REGISTRY = {
    "regional": {
        # addr_state confirmed in 890k schema
        "description": "Geographic leverage risk",
        "dimensions": ["addr_state"],
        "metric_sql": "AVG(dti)",
        "metric_name": "avg_dti",
        "insight_cols": {"key": "addr_state", "metric_label": "avg DTI"},
    },
    "vintage_risk": {
        # issue_d confirmed in 890k schema (was issue_month in OpenIntro version)
        "description": "Temporal delinquency by cohort and grade",
        "dimensions": ["issue_d", "grade"],
        "metric_sql": (
            "SUM(CASE WHEN loan_status LIKE '%Late%' "
            "         OR loan_status = 'Default'      "
            "         OR loan_status = 'Charged Off'  "
            "    THEN 1 ELSE 0 END) * 1.0 / COUNT(*)"
        ),
        "metric_name": "delinquency_rate",
        "insight_cols": {"key": "issue_d", "secondary": "grade", "metric_label": "delinquency rate"},
    },
    "credit_quality": {
        # purpose confirmed in 890k schema (was loan_purpose in OpenIntro version)
        "description": "Credit quality drift by loan purpose (grade-implied risk + derogatory marks)",
        "dimensions": ["purpose"],
        "metric_sql": (
            "AVG("
            "  CASE grade "
            "    WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 "
            "    WHEN 'D' THEN 4 WHEN 'E' THEN 5 WHEN 'F' THEN 6 "
            "    WHEN 'G' THEN 7 ELSE NULL END"
            "  + COALESCE(delinq_2yrs, 0) * 0.5"
            "  + COALESCE(pub_rec_bankruptcies, 0) * 2.0"
            ")"
        ),
        "metric_name": "composite_risk_score",
        "insight_cols": {"key": "purpose", "metric_label": "composite risk score"},
    },
    "yield_analysis": {
        # int_rate confirmed in 890k schema (was interest_rate in OpenIntro version)
        "description": "Yield capture anomalies by employment tenure",
        "dimensions": ["emp_length"],
        "metric_sql": "AVG(int_rate)",
        "metric_name": "avg_int_rate",
        "insight_cols": {"key": "emp_length", "metric_label": "avg interest rate"},
    },
    "utilisation_stress": {
        # home_ownership confirmed in 890k schema (was homeownership in OpenIntro version)
        # revol_util is already a % ratio in this dataset — use directly
        "description": "Credit utilisation stress by homeownership segment",
        "dimensions": ["home_ownership", "grade"],
        "metric_sql": "AVG(revol_util)",
        "metric_name": "avg_revol_util",
        "insight_cols": {"key": "home_ownership", "secondary": "grade", "metric_label": "avg revolving utilisation %"},
    },
    "income_verification": {
        # verification_status confirmed in 890k schema (was verified_income in OpenIntro version)
        "description": "Rate disparity across income verification tiers",
        "dimensions": ["verification_status", "purpose"],
        "metric_sql": "AVG(int_rate)",
        "metric_name": "avg_int_rate",
        "insight_cols": {"key": "verification_status", "secondary": "purpose", "metric_label": "avg interest rate"},
    },
}


class HunterAgent:
    """
    Scans a DuckDB loans table across multiple analytical dimensions,
    flagging statistically anomalous segments using z-score thresholds.

    Parameters
    ----------
    db_path        : Path to the DuckDB database file.
    min_sample_size: Minimum rows per group; smaller groups are excluded to
                     avoid noise from thin segments.
    z_threshold    : Absolute z-score cutoff above which a segment is flagged.
    top_n          : Maximum anomalies returned per dimension.
    dimensions     : Optional list of dimension keys from DIMENSION_REGISTRY
                     to run. Defaults to all registered dimensions.
    """

    def __init__(
        self,
        db_path:         str   = "data/portfolio.db",
        min_sample_size: int   = 100,
        z_threshold:     float = 2.0,
        top_n:           int   = 10,
        dimensions:      list  = None,
        connector=None,         # optional DataConnector (overrides db_path)
        algorithm:       str   = "zscore",   # zscore | iqr | isolation_forest | lof
        config_dimensions: list = None,      # dimensions from config.yaml
    ):
        self.db_path         = db_path
        self.min_sample_size = min_sample_size
        self.z_threshold     = z_threshold
        self.top_n           = top_n
        self.algorithm       = algorithm
        self._connector      = connector    # external DataConnector if provided
        self._con            = None         # internal DuckDB connection

        # Build registry from config if provided, else use hardcoded DIMENSION_REGISTRY
        if config_dimensions:
            self._registry = build_dimension_registry(config_dimensions)
        else:
            self._registry = DIMENSION_REGISTRY

        self.active_dimensions = dimensions or list(self._registry.keys())

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open(self):
        # Use external DataConnector if provided
        if self._connector is not None:
            return self._connector._con
        if self._con is None:
            self._con = duckdb.connect(self.db_path)
        return self._con

    def _close(self):
        # Only close internal connections — external connector manages its own
        if self._connector is None and self._con:
            self._con.close()
            self._con = None

    def _close(self):
        if self._con:
            self._con.close()
            self._con = None

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _get_column_map(self) -> dict:
        """
        Returns a normalised key → actual column name mapping.
        Normalised key = lowercase with underscores removed, so lookups
        are robust to casing and separator differences across datasets.
        """
        con = self._open()
        cols = con.execute("PRAGMA table_info(loans)").df()["name"].tolist()
        return {c.lower().replace("_", ""): c for c in cols}

    def _verify_columns(self, required_cols: list, cmap: dict) -> list[str]:
        """
        Checks that all columns required by a dimension exist in the table.
        Returns a list of missing column names (empty = all present).
        """
        normalised = set(cmap.keys())
        missing = [
            c for c in required_cols
            if c.lower().replace("_", "") not in normalised
        ]
        return missing

    # ------------------------------------------------------------------
    # Core scan engine
    # ------------------------------------------------------------------

    def scan_granularity(
        self,
        dimensions:  list[str],
        metric_sql:  str,
        metric_name: str,
    ) -> pd.DataFrame:
        """
        Groups the data by `dimensions`, computes `metric_sql` per group,
        then runs the configured anomaly detection algorithm.

        For zscore/iqr: uses SQL-level computation for speed.
        For isolation_forest/lof: fetches grouped data then applies sklearn.
        """
        from agents.detectors import get_detector

        dim_str = ", ".join(dimensions)
        table   = self._connector.table_name if self._connector else "loans"

        # Always fetch grouped data first
        group_query = f"""
        SELECT
            {dim_str},
            {metric_sql} AS metric_value,
            COUNT(*)     AS volume
        FROM {table}
        GROUP BY {dim_str}
        HAVING COUNT(*) >= {self.min_sample_size}
        """
        grouped_df = self._open().execute(group_query).df()

        if grouped_df.empty:
            return grouped_df

        # Route to the right detector
        detector = get_detector(
            algorithm       = self.algorithm,
            threshold       = self.z_threshold,
            min_sample_size = self.min_sample_size,
        )
        result = detector.detect(grouped_df, metric_col="metric_value", top_n=self.top_n)
        return result

    # ------------------------------------------------------------------
    # Orchestrated audit
    # ------------------------------------------------------------------

    def run_strategic_audit(self) -> dict[str, pd.DataFrame]:
        """
        Runs all active dimensions from DIMENSION_REGISTRY and returns a
        dict mapping dimension name → anomaly DataFrame.

        Skips any dimension whose required columns are absent in the table,
        logging a warning rather than raising.
        """
        print("🕵️  Hunter Agent: schema introspection → full-dimension audit")
        cmap = self._get_column_map()
        report_cards = {}

        try:
            for dim_key in self.active_dimensions:
                if dim_key not in self._registry:
                    print(f"  ⚠️  Unknown dimension '{dim_key}' — skipping.")
                    continue

                spec = self._registry[dim_key]
                missing = self._verify_columns(spec["dimensions"], cmap)
                if missing:
                    print(
                        f"  ⚠️  Skipping '{dim_key}': "
                        f"missing columns {missing}"
                    )
                    report_cards[dim_key] = pd.DataFrame()
                    continue

                print(f"  🔍 Scanning: {dim_key} — {spec['description']}")
                try:
                    result = self.scan_granularity(
                        dimensions=spec["dimensions"],
                        metric_sql=spec["metric_sql"],
                        metric_name=spec["metric_name"],
                    )
                    report_cards[dim_key] = result
                    status = f"{len(result)} anomalies" if not result.empty else "clean"
                    print(f"     ↳ {status}")
                except Exception as e:
                    print(f"  ❌ Error in '{dim_key}': {e}")
                    report_cards[dim_key] = pd.DataFrame()

        finally:
            self._close()

        return report_cards
