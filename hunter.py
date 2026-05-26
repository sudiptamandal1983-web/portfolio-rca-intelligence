import duckdb
import pandas as pd


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
        db_path: str = "data/portfolio.db",
        min_sample_size: int = 100,
        z_threshold: float = 2.0,
        top_n: int = 10,
        dimensions: list = None,
    ):
        self.db_path = db_path
        self.min_sample_size = min_sample_size
        self.z_threshold = z_threshold
        self.top_n = top_n
        self.active_dimensions = dimensions or list(DIMENSION_REGISTRY.keys())
        self._con = None  # shared connection — opened once per audit

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open(self):
        if self._con is None:
            self._con = duckdb.connect(self.db_path)
        return self._con

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
        dimensions: list[str],
        metric_sql: str,
        metric_name: str,
    ) -> pd.DataFrame:
        """
        Groups the loans table by `dimensions`, computes `metric_sql` per
        group, then z-scores each group against the population of groups.
        Returns only segments whose |z-score| exceeds self.z_threshold,
        ordered by severity.
        """
        dim_str = ", ".join(dimensions)
        query = f"""
        WITH GroupedData AS (
            SELECT
                {dim_str},
                {metric_sql}   AS metric_value,
                COUNT(*)       AS volume
            FROM loans
            GROUP BY {dim_str}
            HAVING COUNT(*) >= {self.min_sample_size}
        ),
        Stats AS (
            SELECT
                AVG(metric_value)    AS avg_val,
                STDDEV(metric_value) AS std_val
            FROM GroupedData
        )
        SELECT
            g.*,
            s.avg_val,
            s.std_val,
            ROUND(
                (g.metric_value - s.avg_val) / NULLIF(s.std_val, 0),
                2
            ) AS z_score
        FROM GroupedData g
        CROSS JOIN Stats s
        WHERE ABS(
            (g.metric_value - s.avg_val) / NULLIF(s.std_val, 0)
        ) > {self.z_threshold}
        ORDER BY ABS(z_score) DESC
        LIMIT {self.top_n}
        """
        return self._open().execute(query).df()

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
                if dim_key not in DIMENSION_REGISTRY:
                    print(f"  ⚠️  Unknown dimension '{dim_key}' — skipping.")
                    continue

                spec = DIMENSION_REGISTRY[dim_key]
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
