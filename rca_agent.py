import json
import os
import textwrap
from typing import Optional

import pandas as pd

from .hunter import DIMENSION_REGISTRY

# ---------------------------------------------------------------------------
# Narrative templates — one per dimension key in DIMENSION_REGISTRY.
# Each template is a callable: (top_row, df, spec) -> str
# This replaces the old elif chain. Adding a dimension = adding one entry here.
# ---------------------------------------------------------------------------

def _fmt(value, pct=False, decimals=2):
    """Format a metric value for display."""
    v = round(float(value), decimals)
    return f"{v}%" if pct else str(v)


NARRATIVE_TEMPLATES = {
    "regional": lambda row, df, spec: (
        f"🚨 REGIONAL ALERT: State '{_col(row, 'addr_state')}' shows abnormal average DTI of "
        f"{_fmt(row['metric_value'])} vs portfolio mean of {_fmt(row['avg_val'])}. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
    "vintage_risk": lambda row, df, spec: (
        f"🚨 VINTAGE RISK: Cohort {_col(row, 'issue_d')} Grade {_col(row, 'grade')} "
        f"has delinquency rate of {_fmt(row['metric_value'] * 100)}% "
        f"vs cohort mean of {_fmt(row['avg_val'] * 100)}%. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
    "credit_quality": lambda row, df, spec: (
        f"🚨 CREDIT QUALITY ALERT: Loan purpose '{_col(row, 'purpose')}' has composite "
        f"risk score of {_fmt(row['metric_value'], decimals=3)} "
        f"(portfolio mean: {_fmt(row['avg_val'], decimals=3)}). "
        f"Score encodes grade, delinquency frequency, and bankruptcy history. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
    "yield_analysis": lambda row, df, spec: (
        f"🚨 YIELD ANOMALY: Employment tenure '{_col(row, 'emp_length')}' is pricing at "
        f"{_fmt(row['metric_value'])}% average interest rate "
        f"vs tenure mean of {_fmt(row['avg_val'])}%. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
    "utilisation_stress": lambda row, df, spec: (
        f"🚨 UTILISATION STRESS: Segment '{_col(row, 'home_ownership')} / Grade {_col(row, 'grade')}' "
        f"shows average revolving utilisation of {_fmt(row['metric_value'])}% "
        f"vs segment mean of {_fmt(row['avg_val'])}%. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
    "income_verification": lambda row, df, spec: (
        f"🚨 VERIFICATION DISPARITY: '{_col(row, 'verification_status')}' borrowers in "
        f"'{_col(row, 'purpose')}' are priced at {_fmt(row['metric_value'])}% "
        f"vs verification-tier mean of {_fmt(row['avg_val'])}%. "
        f"Z-score: {row['z_score']} across {int(row['volume']):,} loans."
    ),
}


def _col(row: pd.Series, name: str) -> str:
    """
    Safe column accessor: looks up `name` in the row index, trying
    exact match first, then case-insensitive partial match.
    Returns '?' if not found rather than raising KeyError.
    """
    if name in row.index:
        return str(row[name])
    # fallback: partial match on lowercased index
    match = next((k for k in row.index if name.lower() in k.lower()), None)
    return str(row[match]) if match else "?"


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def build_llm_prompt(
    dim_key: str,
    df: pd.DataFrame,
    spec: dict,
    enrichments: list = None,
) -> str:
    """
    Builds a structured prompt for an LLM to generate a natural language
    root cause analysis narrative.

    When enrichments (from CorrelationAgent) are provided, co-movement
    context is injected so the LLM can reason about WHY the anomaly
    occurred, not just THAT it occurred.
    """
    top3 = df.head(3).copy()
    keep_cols = spec.get("dimensions", []) + ["metric_value", "avg_val", "volume", "z_score"]
    top3 = top3[[c for c in keep_cols if c in top3.columns]]
    top3 = top3.round(4)
    rows_json = top3.to_dict(orient="records")

    # Build co-movement context block if enrichments are available
    co_movement_block = ""
    if enrichments:
        co_lines = []
        for e in enrichments[:3]:
            hint = e.get("narrative_hint", "")
            direction = e.get("risk_direction", "")
            co_movers = e.get("co_movers", [])[:4]
            co_detail = "; ".join(
                f"{m['label']}={round(m['value'], 3)} (z={m['z_score']:+.2f}, {m['signal']})"
                for m in co_movers
            )
            co_lines.append(
                f"  Segment hint : {hint}\n"
                f"  Co-movers    : {co_detail}\n"
                f"  Risk pattern : {direction}"
            )
        co_movement_block = (
            "\nCO-MOVEMENT CONTEXT (other metrics at the same segment level):\n"
            + "\n---\n".join(co_lines)
        )

    prompt = textwrap.dedent(f"""
        You are a senior credit risk analyst at a consumer lending bank.
        Analyse the anomaly data and co-movement context below.
        Write a 3-4 sentence executive insight suitable for a portfolio review.

        DIMENSION   : {dim_key}
        DESCRIPTION : {spec.get('description', '')}
        METRIC      : {spec.get('metric_name', '')}

        TOP ANOMALIES (z-scored segments, ordered by severity):
        {json.dumps(rows_json, indent=2)}
        {co_movement_block}

        FOCUS RULE: Your narrative must describe ONLY the TOP segment (row index 0 in
        the data above). Other rows are shown for context only — do not narrate them.

        STRICT GROUNDING RULES — failure to follow these will result in rejection:
        1. ONLY reference numbers, segments, and metrics explicitly present in the
           data above. Do NOT introduce any figures, counts, rates, or comparisons
           that are not shown in the data.
        2. Do NOT reference other segments, states, cohorts, or dimensions not
           listed in the data above.
        3. Z-score direction matters:
           - POSITIVE z-score = segment is ABOVE the population average
           - NEGATIVE z-score = segment is BELOW the population average
           - Never reverse this interpretation.
        4. Co-mover signals:
           - "suppressed" or negative z = metric is LOWER than average
           - "elevated" or positive z = metric is HIGHER than average
           - Always check the direction before calling something a risk signal.
        5. Risk pattern alignment:
           - If risk_direction = "improving" -> frame as positive signal, not alarm
           - If risk_direction = "deteriorating" -> frame as risk concern
           - Never use alarm language unless z-score magnitude > 3.0
        6. 3-4 sentences maximum. One causal hypothesis only.
           Plain prose, no bullets, no headers, no emoji unless z > 3.0.
    """).strip()

    return prompt


def inject_macro_context(prompt: str, macro_contexts: dict, dim_key: str) -> str:
    """
    Injects macro narrative into an existing LLM prompt if available
    for the given dimension. Handles both MacroContext objects and dicts.
    """
    contexts = macro_contexts.get(dim_key, [])
    if not contexts:
        return prompt
    top = contexts[0]

    # Handle both MacroContext objects and plain dicts
    if hasattr(top, "regime_label"):
        regime   = top.regime_label or "unknown"
        narrative= top.macro_narrative or ""
        orig     = top.origination_month or "unknown"
        perf     = top.performance_month or "unknown"
    else:
        regime   = top.get("regime_label", "unknown")
        narrative= top.get("macro_narrative", "")
        orig     = top.get("origination_month", "unknown")
        perf     = top.get("performance_month", "unknown")

    if not narrative:
        return prompt

    macro_block = f"""
MACRO CONTEXT (FRED data at origination and performance):
  Regime at origination : {regime}
  Macro narrative       : {narrative}
  Origination month     : {orig}
  Performance month     : {perf}

Additional instruction:
  - Reference the macro regime and how it may have contributed to this anomaly.
  - If the macro environment was benign, note that the anomaly is likely
    idiosyncratic rather than macro-driven.
"""
    return prompt + macro_block


# ---------------------------------------------------------------------------
# RCA Agent
# ---------------------------------------------------------------------------

class RCAAgent:
    """
    Generates root cause analysis narratives for anomalies surfaced by HunterAgent.
    Accepts optional CorrelationAgent enrichments to produce causal narratives.

    Two modes:
      - Template mode (default): fast, deterministic string narratives.
      - LLM mode: pass use_llm=True and set ANTHROPIC_API_KEY in env (or pass
        llm_client directly) to get natural language narratives from Claude.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client=None,
        model: str = "claude-sonnet-4-5",
    ):
        self.use_llm = use_llm
        self.model = model
        self._client = llm_client

        if use_llm and llm_client is None:
            self._client = self._init_llm_client()

    def _init_llm_client(self):
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("  ⚠️  ANTHROPIC_API_KEY not set — falling back to template narratives.")
                self.use_llm = False
                return None
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print(
                "  ⚠️  'anthropic' package not installed — falling back to template narratives.\n"
                "      Run: pip install anthropic"
            )
            self.use_llm = False
            return None

    def _template_narrative(
        self, dim_key: str, df: pd.DataFrame, spec: dict,
        enrichments: list = None,
    ) -> str:
        """Generates a deterministic narrative, appending co-mover summary if available."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        top_row = df.iloc[0]

        template_fn = NARRATIVE_TEMPLATES.get(dim_key)
        base = None
        if template_fn:
            try:
                base = template_fn(top_row, df, spec)
            except Exception:
                pass

        if base is None:
            base = (
                f"🚨 ANOMALY [{dim_key.upper()}]: top segment has metric value "
                f"{_fmt(top_row.get('metric_value', '?'))} "
                f"(mean: {_fmt(top_row.get('avg_val', '?'))}), "
                f"Z-score: {top_row.get('z_score', '?')}."
            )

        # Append co-mover summary if present
        if enrichments:
            top_e = enrichments[0]
            co_movers = top_e.get("co_movers", [])[:3]
            if co_movers:
                co_str = ", ".join(
                    f"{m['label']} ({m['signal']}, z={m['z_score']:+.2f})"
                    for m in co_movers
                )
                risk = top_e.get("risk_direction", "")
                base += (
                    f"\n   Co-movers: {co_str}."
                    f" Overall pattern: {risk}."
                )

        return base

    def _llm_narrative(
        self, dim_key: str, df: pd.DataFrame, spec: dict,
        enrichments: list = None,
        macro_contexts: dict = None,
    ) -> Optional[str]:
        """Calls the LLM with co-movement and macro context included in the prompt."""
        try:
            prompt = build_llm_prompt(dim_key, df, spec, enrichments)
            if macro_contexts:
                prompt = inject_macro_context(prompt, macro_contexts, dim_key)
            response = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"  ⚠️  LLM call failed for '{dim_key}': {e} — using template.")
            return None

    def analyze_findings(
        self,
        df: pd.DataFrame,
        dim_key: str,
        enrichments: list = None,
        macro_contexts: dict = None,
    ) -> tuple:
        """
        Generates a narrative for the top anomaly in df.
        Returns (narrative, llm_generated) tuple.
        Optionally accepts enrichments from CorrelationAgent for causal context.
        """
        if df is None or df.empty:
            return None, False

        spec = DIMENSION_REGISTRY.get(dim_key, {})

        if self.use_llm and self._client:
            narrative = self._llm_narrative(dim_key, df, spec, enrichments, macro_contexts)
            if narrative:
                return narrative, True

        return self._template_narrative(dim_key, df, spec, enrichments), False

    def analyze_all(
        self,
        report_cards: dict,
        enriched_report: dict = None,
        macro_contexts: dict = None,
    ) -> list:
        """
        Runs analyze_findings over the full report_cards dict.
        Passes correlation enrichments to each dimension if enriched_report provided.

        Returns a ranked list of insight dicts sorted by |z_score| descending.
        Each dict contains: dimension, narrative, top_z, anomaly_count,
                            co_movers, risk_direction, raw.
        """
        enriched_report = enriched_report or {}
        insights = []

        for dim_key, df in report_cards.items():
            if df is None or df.empty:
                continue

            enrichments = enriched_report.get(dim_key, [])
            narrative, llm_used = self.analyze_findings(df, dim_key, enrichments, macro_contexts)

            if narrative:
                df_norm = df.copy()
                df_norm.columns = df_norm.columns.str.lower().str.strip()
                top_enrichment = enrichments[0] if enrichments else {}
                insights.append({
                    "dimension":      dim_key,
                    "narrative":      narrative,
                    "llm_generated":  llm_used,
                    "top_z":          abs(float(df_norm.iloc[0].get("z_score", 0))),
                    "anomaly_count":  len(df_norm),
                    "co_movers":      top_enrichment.get("co_movers", []),
                    "risk_direction": top_enrichment.get("risk_direction", "unknown"),
                    "raw":            df_norm.to_dict(orient="records"),
                })

        return sorted(insights, key=lambda x: x["top_z"], reverse=True)