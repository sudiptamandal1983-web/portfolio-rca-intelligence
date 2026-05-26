"""
correlation_agent.py — Cross-metric co-movement analysis.

When HunterAgent flags a segment as anomalous on one metric, this agent
checks that same segment across all other metrics to surface co-movement.

The output answers: "X changed — what else moved with it, and how strongly?"

Architecture:
    Hunter flags:  addr_state=NV  avg_dti=42.1  z=+2.8
                        ↓
    CorrelationAgent queries NV across all METRIC_REGISTRY entries:
        int_rate       → 18.9%  z=+2.1  ← co-moving UP
        revol_util     → 84.3%  z=+1.8  ← co-moving UP
        delinquency    → 12.4%  z=+2.4  ← co-moving UP
        annual_inc     → 51200  z=-1.2  ← slightly low (not significant)
                        ↓
    RCAAgent receives enriched finding with co-movers attached
"""

import duckdb
import pandas as pd
from typing import Optional

from .hunter import DIMENSION_REGISTRY


# ---------------------------------------------------------------------------
# Metric registry — all scalar metrics we can probe at any dimension level.
# These are the "Y" variables we check whenever Hunter flags an "X".
# Keyed by a short name; sql must produce a single scalar per group.
# ---------------------------------------------------------------------------
METRIC_REGISTRY = {
    "avg_dti": {
        "label": "Avg DTI",
        "sql": "AVG(dti)",
        "format": ".1f",
        "direction": "higher_is_worse",
    },
    "avg_int_rate": {
        "label": "Avg interest rate",
        "sql": "AVG(int_rate)",
        "format": ".2f",
        "direction": "higher_is_worse",
    },
    "delinquency_rate": {
        "label": "Delinquency rate",
        "sql": (
            "SUM(CASE WHEN loan_status LIKE '%Late%' "
            "         OR loan_status = 'Default' "
            "         OR loan_status = 'Charged Off' "
            "    THEN 1 ELSE 0 END) * 1.0 / COUNT(*)"
        ),
        "format": ".2%",
        "direction": "higher_is_worse",
    },
    "avg_revol_util": {
        "label": "Avg revolving utilisation %",
        "sql": "AVG(revol_util)",
        "format": ".1f",
        "direction": "higher_is_worse",
    },
    "avg_annual_inc": {
        "label": "Avg annual income",
        "sql": "AVG(annual_inc)",
        "format": ".0f",
        "direction": "lower_is_worse",
    },
    "avg_loan_amnt": {
        "label": "Avg loan amount",
        "sql": "AVG(loan_amnt)",
        "format": ".0f",
        "direction": "neutral",
    },
    "avg_grade_score": {
        "label": "Avg grade score (A=1 … G=7)",
        "sql": (
            "AVG(CASE grade "
            "  WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 "
            "  WHEN 'D' THEN 4 WHEN 'E' THEN 5 WHEN 'F' THEN 6 "
            "  WHEN 'G' THEN 7 ELSE NULL END)"
        ),
        "format": ".2f",
        "direction": "higher_is_worse",
    },
    "avg_delinq_2yrs": {
        "label": "Avg delinquencies (2yr)",
        "sql": "AVG(delinq_2yrs)",
        "format": ".2f",
        "direction": "higher_is_worse",
    },
    "avg_open_acc": {
        "label": "Avg open accounts",
        "sql": "AVG(open_acc)",
        "format": ".1f",
        "direction": "neutral",
    },
    "avg_pub_rec": {
        "label": "Avg public records",
        "sql": "AVG(pub_rec)",
        "format": ".2f",
        "direction": "higher_is_worse",
    },
}

# Significance threshold for co-movers — lower than Hunter's 2.0
# to catch directional signals that don't reach full anomaly status
CO_MOVE_Z_THRESHOLD = 1.5


class CorrelationAgent:
    """
    For each anomalous segment surfaced by HunterAgent, queries all metrics
    in METRIC_REGISTRY at the same dimension level and computes z-scores to
    identify which metrics co-move with the flagged anomaly.

    Parameters
    ----------
    db_path          : Path to the DuckDB database file.
    min_sample_size  : Minimum group size (same as HunterAgent for consistency).
    co_move_threshold: Z-score threshold for a co-mover to be reported.
                       Intentionally lower than Hunter's anomaly threshold —
                       we want directional signals, not just confirmed anomalies.
    """

    def __init__(
        self,
        db_path: str = "data/portfolio.db",
        min_sample_size: int = 100,
        co_move_threshold: float = CO_MOVE_Z_THRESHOLD,
    ):
        self.db_path = db_path
        self.min_sample_size = min_sample_size
        self.co_move_threshold = co_move_threshold
        self._con = None

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
    # Core: probe a single metric at a fixed dimension segment
    # ------------------------------------------------------------------

    def _probe_metric(
        self,
        dimension_cols: list[str],
        segment_filter: dict,
        metric_key: str,
    ) -> Optional[dict]:
        """
        For a given segment (e.g. addr_state='NV'), compute metric_key
        for that segment AND compute its z-score against all peers
        at the same dimension level.

        Returns a dict with value, population mean, z-score, and volume,
        or None if the segment doesn't meet min_sample_size.
        """
        metric = METRIC_REGISTRY[metric_key]
        dim_str = ", ".join(dimension_cols)

        # Build WHERE clause for the specific segment
        where_parts = [
            f"{col} = '{val}'" for col, val in segment_filter.items()
        ]
        where_clause = " AND ".join(where_parts)

        query = f"""
        WITH AllGroups AS (
            SELECT
                {dim_str},
                {metric['sql']} AS metric_value,
                COUNT(*)        AS volume
            FROM loans
            GROUP BY {dim_str}
            HAVING COUNT(*) >= {self.min_sample_size}
        ),
        Stats AS (
            SELECT
                AVG(metric_value)    AS avg_val,
                STDDEV(metric_value) AS std_val,
                COUNT(*)             AS n_groups
            FROM AllGroups
        ),
        Target AS (
            SELECT
                {dim_str},
                {metric['sql']} AS metric_value,
                COUNT(*)        AS volume
            FROM loans
            WHERE {where_clause}
            GROUP BY {dim_str}
            HAVING COUNT(*) >= {self.min_sample_size}
        )
        SELECT
            t.metric_value,
            t.volume,
            s.avg_val,
            s.std_val,
            s.n_groups,
            ROUND(
                (t.metric_value - s.avg_val) / NULLIF(s.std_val, 0),
                3
            ) AS z_score
        FROM Target t
        CROSS JOIN Stats s
        """

        try:
            result = self._open().execute(query).df()
            if result.empty:
                return None
            row = result.iloc[0]
            return {
                "metric_key":   metric_key,
                "label":        metric["label"],
                "value":        float(row["metric_value"]),
                "avg_val":      float(row["avg_val"]),
                "z_score":      float(row["z_score"]),
                "volume":       int(row["volume"]),
                "n_groups":     int(row["n_groups"]),
                "direction":    metric["direction"],
                "format":       metric["format"],
            }
        except Exception as e:
            return None

    # ------------------------------------------------------------------
    # Classify co-movement signal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(z: float, direction: str) -> str:
        """
        Returns a human-readable signal label given z-score and metric direction.

        direction='higher_is_worse': positive z = risk increasing
        direction='lower_is_worse':  negative z = risk increasing
        direction='neutral':         just magnitude
        """
        abs_z = abs(z)
        if abs_z < 0.5:
            return "stable"

        if direction == "higher_is_worse":
            if z > 0:
                strength = "strongly elevated" if abs_z >= 2.0 else "elevated"
                return f"⬆ {strength}"
            else:
                strength = "strongly suppressed" if abs_z >= 2.0 else "suppressed"
                return f"⬇ {strength}"

        elif direction == "lower_is_worse":
            if z < 0:
                strength = "strongly below norm" if abs_z >= 2.0 else "below norm"
                return f"⬇ {strength} (risk signal)"
            else:
                return f"⬆ above norm"

        else:  # neutral
            if z > 0:
                return f"⬆ above norm"
            else:
                return f"⬇ below norm"

    # ------------------------------------------------------------------
    # Public: enrich a single anomaly with co-movers
    # ------------------------------------------------------------------

    def enrich_anomaly(
        self,
        dim_key: str,
        anomaly_row: pd.Series,
        primary_metric_key: str,
    ) -> dict:
        """
        Takes one anomalous segment row from HunterAgent output and probes
        all METRIC_REGISTRY metrics at the same dimension level.

        Returns an enriched dict:
        {
            "segment":        {"addr_state": "NV"},
            "primary":        {metric_key, value, z_score, ...},
            "co_movers":      [{metric_key, label, value, z_score, signal}, ...],
            "co_mover_count": int,
            "risk_direction": "deteriorating" | "improving" | "mixed" | "isolated",
            "narrative_hint": str   ← one-line summary for RCA/LLM
        }
        """
        spec = DIMENSION_REGISTRY.get(dim_key, {})
        dimension_cols = spec.get("dimensions", [])

        # Build segment filter from the anomaly row
        anomaly_row.index = anomaly_row.index.str.lower().str.strip()
        segment_filter = {}
        for col in dimension_cols:
            col_lower = col.lower()
            if col_lower in anomaly_row.index:
                segment_filter[col] = anomaly_row[col_lower]

        if not segment_filter:
            return {}

        # Probe all metrics except the primary (already known)
        co_movers = []
        for metric_key, metric_spec in METRIC_REGISTRY.items():
            if metric_key == primary_metric_key:
                continue
            result = self._probe_metric(dimension_cols, segment_filter, metric_key)
            if result is None:
                continue
            if abs(result["z_score"]) >= self.co_move_threshold:
                result["signal"] = self._classify(
                    result["z_score"], result["direction"]
                )
                co_movers.append(result)

        # Sort by absolute z-score descending
        co_movers.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        # Classify overall risk direction
        risk_direction = self._risk_direction(co_movers)

        # Build a one-line narrative hint for RCA/LLM context
        narrative_hint = self._build_narrative_hint(
            segment_filter, primary_metric_key, anomaly_row, co_movers, risk_direction
        )

        return {
            "segment":        segment_filter,
            "primary":        {
                "metric_key": primary_metric_key,
                "value":      float(anomaly_row.get("metric_value", 0)),
                "z_score":    float(anomaly_row.get("z_score", 0)),
            },
            "co_movers":      co_movers,
            "co_mover_count": len(co_movers),
            "risk_direction": risk_direction,
            "narrative_hint": narrative_hint,
        }

    # ------------------------------------------------------------------
    # Enrich all anomalies from a full audit run
    # ------------------------------------------------------------------

    def enrich_report(
        self,
        report_cards: dict[str, pd.DataFrame],
        top_n_per_dimension: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Runs enrich_anomaly over the top N anomalies in each dimension.

        Parameters
        ----------
        report_cards           : Output of HunterAgent.run_strategic_audit()
        top_n_per_dimension    : How many top anomalies per dimension to enrich.
                                 Keep low (2-3) — each enrichment is a DB round trip
                                 per metric in METRIC_REGISTRY.

        Returns a dict: dimension_key → [enriched_anomaly_dict, ...]
        """
        print("🔗  Correlation Agent: probing co-movement for flagged segments...")
        enriched = {}

        try:
            for dim_key, df in report_cards.items():
                if df is None or df.empty:
                    continue

                spec = DIMENSION_REGISTRY.get(dim_key, {})
                primary_metric = spec.get("metric_name", "unknown")
                dim_enriched = []

                top_rows = df.head(top_n_per_dimension)
                print(f"  🔗 {dim_key}: enriching top {len(top_rows)} anomalies")

                for _, row in top_rows.iterrows():
                    enrichment = self.enrich_anomaly(dim_key, row, primary_metric)
                    if enrichment:
                        dim_enriched.append(enrichment)

                enriched[dim_key] = dim_enriched

                # Print a quick summary to terminal
                for e in dim_enriched:
                    seg_str = ", ".join(f"{k}={v}" for k, v in e["segment"].items())
                    print(
                        f"     ↳ [{seg_str}] "
                        f"primary z={e['primary']['z_score']:+.2f} | "
                        f"{e['co_mover_count']} co-movers | "
                        f"risk: {e['risk_direction']}"
                    )

        finally:
            self._close()

        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_direction(co_movers: list[dict]) -> str:
        """
        Summarises whether co-movers collectively point toward
        deterioration, improvement, mixed signals, or isolated anomaly.
        """
        if not co_movers:
            return "isolated"

        worsening = sum(
            1 for m in co_movers
            if (m["direction"] == "higher_is_worse" and m["z_score"] > 0)
            or (m["direction"] == "lower_is_worse"  and m["z_score"] < 0)
        )
        improving = sum(
            1 for m in co_movers
            if (m["direction"] == "higher_is_worse" and m["z_score"] < 0)
            or (m["direction"] == "lower_is_worse"  and m["z_score"] > 0)
        )

        total = len(co_movers)
        if worsening / total >= 0.7:
            return "deteriorating"
        elif improving / total >= 0.7:
            return "improving"
        else:
            return "mixed"

    @staticmethod
    def _build_narrative_hint(
        segment_filter: dict,
        primary_metric_key: str,
        anomaly_row: pd.Series,
        co_movers: list[dict],
        risk_direction: str,
    ) -> str:
        """
        Builds a compact one-line narrative hint summarising the co-movement
        pattern. Used as context injection for the LLM prompt builder.
        """
        seg_str = ", ".join(f"{k}={v}" for k, v in segment_filter.items())
        primary_z = float(anomaly_row.get("z_score", 0))

        if not co_movers:
            return (
                f"Segment [{seg_str}] is anomalous on {primary_metric_key} "
                f"(z={primary_z:+.2f}) with no significant co-movers — "
                f"potentially an isolated pricing or data artefact."
            )

        top_co = co_movers[:3]
        co_str = "; ".join(
            f"{m['label']} {m['signal']} (z={m['z_score']:+.2f})"
            for m in top_co
        )

        direction_phrase = {
            "deteriorating": "consistent with broad credit deterioration",
            "improving":     "consistent with improving credit quality",
            "mixed":         "showing mixed signals across risk dimensions",
            "isolated":      "appearing isolated to this metric",
        }.get(risk_direction, "")

        return (
            f"Segment [{seg_str}] flagged on {primary_metric_key} "
            f"(z={primary_z:+.2f}). Co-movers: {co_str}. "
            f"Overall pattern {direction_phrase}."
        )
