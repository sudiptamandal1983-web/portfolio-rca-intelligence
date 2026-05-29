"""
orchestrator.py — Multi-agent pipeline coordinator.

Owns the message bus. Runs agents in sequence, passes typed messages
between them, tracks status, handles failures gracefully, and writes
the final PipelineResult.

Usage:
    from agents import Orchestrator
    result = Orchestrator(config).run()
    print(result.summary())
"""

import json
import os
import uuid
import traceback
from datetime import datetime
from typing import Optional

from .hunter import HunterAgent, DIMENSION_REGISTRY
from .correlation_agent import CorrelationAgent
from .rca_agent import RCAAgent
from .eval_agent import EvaluationAgent, EvalResult
from .macro_agent import MacroAgent, MacroContext
from .email_agent import EmailAgent
from .messages import (
    AnomalyReport, EnrichedReport, InsightBundle,
    PipelineResult, Severity, RecipientGroup,
    CoMover, RiskDirection,
)


class OrchestratorConfig:
    """
    Single place to configure the full pipeline.
    Pass to Orchestrator() instead of juggling CLI args.
    """
    def __init__(
        self,
        db_path:              str   = "data/portfolio.db",
        data_source:          str   = "duckdb",
        data_table:           str   = "loans",
        report_dir:           str   = "reports",
        dimensions:           Optional[list[str]] = None,
        config_dimensions:    Optional[list[dict]] = None,
        algorithm:            str   = "zscore",
        min_sample_size:      int   = 100,
        z_threshold:          float = 2.0,
        co_move_threshold:    float = 1.5,
        top_n_anomalies:      int   = 10,
        top_n_enrichments:    int   = 3,
        use_macro:            bool  = True,
        fred_api_key:         Optional[str] = None,
        use_llm:              bool  = False,
        llm_model:            str   = "claude-sonnet-4-5",
        llm_max_tokens:       int   = 400,
        use_eval:             bool  = False,
        eval_model:           str   = "claude-haiku-4-5-20251001",
        eval_max_tokens:      int   = 200,
        eval_threshold:       float = 3.5,
        eval_fail_open:       bool  = True,
        send_email:           bool  = False,
        email_dry_run:        bool  = False,
        gmail_user:           Optional[str] = None,
        gmail_app_password:   Optional[str] = None,
        recipient_groups:     Optional[list[RecipientGroup]] = None,
    ):
        self.db_path            = db_path
        self.data_source        = data_source
        self.data_table         = data_table
        self.report_dir         = report_dir
        self.dimensions         = dimensions
        self.config_dimensions  = config_dimensions
        self.algorithm          = algorithm
        self.min_sample_size    = min_sample_size
        self.z_threshold        = z_threshold
        self.co_move_threshold  = co_move_threshold
        self.top_n_anomalies    = top_n_anomalies
        self.top_n_enrichments  = top_n_enrichments
        self.use_llm            = use_llm
        self.llm_model          = llm_model
        self.llm_max_tokens     = llm_max_tokens
        self.use_macro          = use_macro
        self.fred_api_key       = fred_api_key
        self.use_eval           = use_eval
        self.eval_model         = eval_model
        self.eval_max_tokens    = eval_max_tokens
        self.eval_threshold     = eval_threshold
        self.eval_fail_open     = eval_fail_open
        self.send_email         = send_email
        self.email_dry_run      = email_dry_run
        self.gmail_user         = gmail_user
        self.gmail_app_password = gmail_app_password
        self.recipient_groups   = recipient_groups


SEP = "═" * 60


class Orchestrator:
    """
    Coordinates the full RCA pipeline:

        HunterAgent → CorrelationAgent → RCAAgent → EmailAgent

    Each agent receives a typed message from the previous agent.
    If an agent fails, the pipeline continues with degraded output
    rather than crashing — partial insights are better than no insights.

    Parameters
    ----------
    config : OrchestratorConfig instance.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config   = config or OrchestratorConfig()
        self.run_id   = str(uuid.uuid4())[:8]
        self._statuses: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        started_at = datetime.now()
        self._print_banner(started_at)

        bundles:  list[InsightBundle]  = []
        receipts = []

        # ── Stage 1: Hunt ──────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 1 / 4 — Hunter Agent")
        print(SEP)
        report_cards = self._run_hunter()

        # ── Stage 2: Correlate ─────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 2 / 4 — Correlation Agent")
        print(SEP)
        enriched_map = self._run_correlator(report_cards)

        # ── Stage 2.5: Macro ───────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 2.5 / 4 — Macro Agent")
        print(SEP)
        macro_contexts = self._run_macro(report_cards)
        self._macro_contexts = macro_contexts  # store for report writer

        # ── Stage 3: RCA ───────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 3 / 4 — RCA Agent")
        print(SEP)
        bundles = self._run_rca(report_cards, enriched_map, macro_contexts)

        # ── Stage 3.5: Evaluate ────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 3.5 / 4 — Evaluation Agent")
        print(SEP)
        bundles, eval_results = self._run_eval(bundles)

        # ── Stage 4: Email ─────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  Stage 4 / 4 — Email Agent")
        print(SEP)
        receipts = self._run_email(bundles)

        # ── Finalise ───────────────────────────────────────────────────
        completed_at = datetime.now()
        report_path  = self._write_report(bundles, receipts, started_at, completed_at)

        result = PipelineResult(
            run_id         = self.run_id,
            started_at     = started_at,
            completed_at   = completed_at,
            db_path        = self.config.db_path,
            bundles        = bundles,
            receipts       = receipts,
            agent_statuses = self._statuses,
            report_path    = report_path,
        )

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Stage runners — each wraps an agent and converts to/from messages
    # ------------------------------------------------------------------

    def _run_hunter(self) -> dict:
        try:
            from agents.data_connector import DataConnector
            connector = DataConnector.from_config({
                "source":     self.config.data_source,
                "path":       self.config.db_path,
                "table":      self.config.data_table,
                "table_name": self.config.data_table,
            })
            hunter = HunterAgent(
                db_path           = self.config.db_path,
                min_sample_size   = self.config.min_sample_size,
                z_threshold       = self.config.z_threshold,
                top_n             = self.config.top_n_anomalies,
                dimensions        = self.config.dimensions,
                connector         = connector,
                algorithm         = self.config.algorithm,
                config_dimensions = self.config.config_dimensions,
            )
            report_cards = hunter.run_strategic_audit()
            n_flagged = sum(1 for df in report_cards.values() if df is not None and not df.empty)
            self._statuses["hunter"] = f"ok — {n_flagged} dimensions flagged"
            return report_cards
        except Exception as e:
            self._statuses["hunter"] = f"error: {e}"
            print(f"  ❌  Hunter Agent failed: {e}")
            traceback.print_exc()
            return {}

    def _run_correlator(self, report_cards: dict) -> dict:
        """
        Runs CorrelationAgent and converts raw dict output to
        EnrichedReport messages keyed by dimension.
        """
        if not report_cards:
            self._statuses["correlation"] = "skipped — no hunter output"
            return {}
        try:
            correlator = CorrelationAgent(
                db_path           = self.config.db_path,
                min_sample_size   = self.config.min_sample_size,
                co_move_threshold = self.config.co_move_threshold,
            )
            raw_enriched = correlator.enrich_report(
                report_cards,
                top_n_per_dimension=self.config.top_n_enrichments,
            )

            # Convert raw dicts → EnrichedReport messages
            enriched_map: dict[str, list[EnrichedReport]] = {}
            for dim_key, raw_list in raw_enriched.items():
                spec    = DIMENSION_REGISTRY.get(dim_key, {})
                reports = []
                for raw in raw_list:
                    # Build the AnomalyReport from the raw segment dict
                    seg = raw.get("segment", {})
                    primary = raw.get("primary", {})
                    z = abs(float(primary.get("z_score", 0)))
                    severity = (
                        Severity.CRITICAL if z >= 3.0 else
                        Severity.WARNING  if z >= 2.0 else
                        Severity.INFO
                    )
                    anomaly = AnomalyReport(
                        dimension     = dim_key,
                        description   = spec.get("description", ""),
                        segment       = seg,
                        metric_name   = spec.get("metric_name", ""),
                        metric_value  = float(primary.get("value", 0)),
                        metric_mean   = 0.0,    # not available from enrichment
                        z_score       = float(primary.get("z_score", 0)),
                        volume        = 0,
                        anomaly_count = 0,
                        severity      = severity,
                        raw_df        = [],
                    )
                    co_movers = [
                        CoMover(
                            metric_key = m["metric_key"],
                            label      = m["label"],
                            value      = float(m["value"]),
                            avg_val    = float(m["avg_val"]),
                            z_score    = float(m["z_score"]),
                            signal     = m["signal"],
                            direction  = m["direction"],
                        )
                        for m in raw.get("co_movers", [])
                    ]
                    direction_str = raw.get("risk_direction", "unknown")
                    try:
                        risk_direction = RiskDirection(direction_str)
                    except ValueError:
                        risk_direction = RiskDirection.UNKNOWN

                    reports.append(EnrichedReport(
                        anomaly        = anomaly,
                        co_movers      = co_movers,
                        risk_direction = risk_direction,
                        narrative_hint = raw.get("narrative_hint", ""),
                    ))
                enriched_map[dim_key] = reports

            n = sum(len(v) for v in enriched_map.values())
            self._statuses["correlation"] = f"ok — {n} segments enriched"
            return enriched_map

        except Exception as e:
            self._statuses["correlation"] = f"error: {e}"
            print(f"  ❌  Correlation Agent failed: {e}")
            traceback.print_exc()
            return {}

    def _run_rca(
        self,
        report_cards: dict,
        enriched_map: dict,
        macro_contexts: dict = None,
    ) -> list[InsightBundle]:
        """
        Runs RCAAgent, merges with enrichment, and wraps output
        into InsightBundle messages.
        """
        if not report_cards:
            self._statuses["rca"] = "skipped — no hunter output"
            return []
        try:
            rca = RCAAgent(
                use_llm = self.config.use_llm,
                model   = self.config.llm_model,
            )

            # Convert enriched_map back to the raw dict format RCAAgent expects
            raw_enriched_for_rca: dict[str, list[dict]] = {}
            for dim_key, enriched_list in enriched_map.items():
                raw_enriched_for_rca[dim_key] = [
                    {
                        "co_movers": [
                            {
                                "metric_key": c.metric_key,
                                "label":      c.label,
                                "value":      c.value,
                                "avg_val":    c.avg_val,
                                "z_score":    c.z_score,
                                "signal":     c.signal,
                                "direction":  c.direction,
                            }
                            for c in e.co_movers
                        ],
                        "risk_direction": e.risk_direction.value,
                        "narrative_hint": e.narrative_hint,
                    }
                    for e in enriched_list
                ]

            insights = rca.analyze_all(report_cards, raw_enriched_for_rca, macro_contexts or {})

            # Wrap each insight into an InsightBundle
            bundles: list[InsightBundle] = []
            for insight in insights:
                dim_key     = insight["dimension"]
                spec        = DIMENSION_REGISTRY.get(dim_key, {})
                enriched_list = enriched_map.get(dim_key, [])
                top_enriched  = enriched_list[0] if enriched_list else None

                # Build AnomalyReport from the RCA insight dict
                raw = insight.get("raw", [])
                top = raw[0] if raw else {}
                z   = abs(insight.get("top_z", 0))
                severity = (
                    Severity.CRITICAL if z >= 3.0 else
                    Severity.WARNING  if z >= 2.0 else
                    Severity.INFO
                )
                dim_cols = spec.get("dimensions", [])
                segment  = {k: v for k, v in top.items() if k in dim_cols}

                anomaly = AnomalyReport(
                    dimension     = dim_key,
                    description   = spec.get("description", ""),
                    segment       = segment,
                    metric_name   = spec.get("metric_name", ""),
                    metric_value  = float(top.get("metric_value", 0)),
                    metric_mean   = float(top.get("avg_val", 0)),
                    z_score       = float(top.get("z_score", 0)),
                    volume        = int(top.get("volume", 0)),
                    anomaly_count = insight.get("anomaly_count", 0),
                    severity      = severity,
                    raw_df        = raw,
                )

                # Use existing EnrichedReport or build a minimal one
                if top_enriched:
                    enriched = EnrichedReport(
                        anomaly        = anomaly,
                        co_movers      = top_enriched.co_movers,
                        risk_direction = top_enriched.risk_direction,
                        narrative_hint = top_enriched.narrative_hint,
                    )
                else:
                    enriched = EnrichedReport(
                        anomaly        = anomaly,
                        co_movers      = [],
                        risk_direction = RiskDirection.UNKNOWN,
                        narrative_hint = "",
                    )

                bundles.append(InsightBundle(
                    enriched      = enriched,
                    narrative     = insight["narrative"],
                    llm_generated = insight.get("llm_generated", False),
                ))

            self._statuses["rca"] = f"ok — {len(bundles)} bundles generated"
            return bundles

        except Exception as e:
            self._statuses["rca"] = f"error: {e}"
            print(f"  ❌  RCA Agent failed: {e}")
            traceback.print_exc()
            return []



    def _run_macro(self, report_cards: dict) -> dict:
        """Runs MacroAgent to enrich anomaly segments with FRED macro context."""
        if not self.config.use_macro:
            print("  ⏭️   Macro enrichment disabled.")
            self._statuses["macro"] = "skipped — disabled"
            return {}
        try:
            macro_agent = MacroAgent(
                db_path  = self.config.db_path,
                api_key  = self.config.fred_api_key or os.getenv("FRED_API_KEY", ""),
            )
            macro_contexts = macro_agent.enrich_report(
                report_cards, DIMENSION_REGISTRY, top_n=3
            )
            n = sum(len(v) for v in macro_contexts.values())
            self._statuses["macro"] = f"ok — {n} segments enriched with macro context"
            return macro_contexts
        except Exception as e:
            self._statuses["macro"] = f"error: {e}"
            print(f"  ❌  Macro Agent failed: {e}")
            traceback.print_exc()
            return {}

    def _run_eval(
        self,
        bundles: list[InsightBundle],
    ) -> tuple[list[InsightBundle], list]:
        """
        Runs EvaluationAgent on all bundles.
        If use_eval=False or use_llm=False, passes through immediately.
        Only LLM-generated narratives need evaluation.
        """
        # Skip eval if LLM is off — templates don't need scoring
        if not self.config.use_llm or not self.config.use_eval:
            mode = "llm off" if not self.config.use_llm else "eval disabled"
            print(f"  ⏭️   Evaluation skipped — {mode}.")
            self._statuses["eval"] = f"skipped — {mode}"
            return bundles, []

        try:
            evaluator = EvaluationAgent(
                pass_threshold = self.config.eval_threshold,
                api_key        = os.getenv("ANTHROPIC_API_KEY", ""),
                model          = self.config.eval_model,
                fail_open      = self.config.eval_fail_open,
            )
            evaluated_bundles, eval_results = evaluator.evaluate_all(bundles)

            passed   = sum(1 for r in eval_results if r.passed)
            fallback = sum(1 for r in eval_results if not r.passed)
            self._statuses["eval"] = (
                f"ok — {passed} passed, {fallback} fell back to template"
            )
            return evaluated_bundles, eval_results

        except Exception as e:
            self._statuses["eval"] = f"error: {e}"
            print(f"  ❌  Evaluation Agent failed: {e} — using original narratives")
            traceback.print_exc()
            return bundles, []

    def _run_email(self, bundles: list[InsightBundle]) -> list:
        if not self.config.send_email and not self.config.email_dry_run:
            print("  ⏭️   Email delivery disabled. "
                  "Pass --email (or --email-dry-run) to enable.")
            self._statuses["email"] = "skipped — disabled"
            return []
        try:
            email_agent = EmailAgent(
                gmail_user         = self.config.gmail_user,
                gmail_app_password = self.config.gmail_app_password,
                recipient_groups   = self.config.recipient_groups,
                dry_run            = self.config.email_dry_run,
            )
            receipts = email_agent.run(bundles)
            sent  = sum(1 for r in receipts if r.status.value == "sent")
            failed = sum(1 for r in receipts if r.status.value == "failed")
            self._statuses["email"] = f"ok — {sent} sent, {failed} failed"
            return receipts
        except Exception as e:
            self._statuses["email"] = f"error: {e}"
            print(f"  ❌  Email Agent failed: {e}")
            traceback.print_exc()
            return []

    # ------------------------------------------------------------------
    # Report writer
    # ------------------------------------------------------------------


    def _get_macro_narrative(self, dimension: str) -> str:
        """Gets macro narrative for a dimension from stored macro contexts."""
        contexts = getattr(self, "_macro_contexts", {}).get(dimension, [])
        if contexts and hasattr(contexts[0], "macro_narrative"):
            return contexts[0].macro_narrative
        return ""

    def _get_macro_regime(self, dimension: str) -> str:
        """Gets macro regime label for a dimension from stored macro contexts."""
        contexts = getattr(self, "_macro_contexts", {}).get(dimension, [])
        if contexts and hasattr(contexts[0], "regime_label"):
            return contexts[0].regime_label
        return ""

    def _write_report(
        self,
        bundles:      list[InsightBundle],
        receipts:     list,
        started_at:   datetime,
        completed_at: datetime,
    ) -> str:
        import os
        os.makedirs(self.config.report_dir, exist_ok=True)
        ts   = started_at.strftime("%Y%m%d_%H%M%S")
        path = f"{self.config.report_dir}/audit_{ts}.json"

        report = {
            "run_id":       self.run_id,
            "generated_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_s":   (completed_at - started_at).total_seconds(),
            "database":     self.config.db_path,
            "llm_mode":     self.config.use_llm,
            "z_threshold":  self.config.z_threshold,
            "agent_statuses": self._statuses,
            "insights": [
                {
                    "dimension":      b.dimension,
                    "severity":       b.severity.value,
                    "top_z":          b.top_z,
                    "segment":        b.enriched.anomaly.segment,
                    "metric_name":    b.enriched.anomaly.metric_name,
                    "metric_value":   b.enriched.anomaly.metric_value,
                    "metric_mean":    b.enriched.anomaly.metric_mean,
                    "volume":         b.enriched.anomaly.volume,
                    "risk_direction": b.risk_direction.value,
                    "narrative":      b.narrative,
                    "llm_generated":  b.llm_generated,
                    "macro_narrative": self._get_macro_narrative(b.dimension),
                    "regime_label":    self._get_macro_regime(b.dimension),
                    "co_movers": [
                        {
                            "label":   c.label,
                            "value":   c.value,
                            "z_score": c.z_score,
                            "signal":  c.signal,
                        }
                        for c in b.enriched.top_co_movers
                    ],
                    "raw": b.enriched.anomaly.raw_df,
                }
                for b in bundles
            ],
            "email_receipts": [
                {
                    "recipient":     r.recipient,
                    "status":        r.status.value,
                    "insights_sent": r.insights_sent,
                    "error":         r.error,
                }
                for r in receipts
            ],
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  📄  Report → {path}")
        return path

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def _print_banner(self, started_at: datetime):
        print(f"\n{SEP}")
        print("  🏦  Banking Portfolio RCA Pipeline")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Started  : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Database : {self.config.db_path}")
        print(f"  Algorithm: {self.config.algorithm}")
        print(f"  LLM mode : {'on' if self.config.use_llm else 'off'}")
        if self.config.use_llm:
            print(f"  LLM model: {self.config.llm_model}")
            print(f"  Eval model: {self.config.eval_model}")
        print(f"  Email    : {'dry-run' if self.config.email_dry_run else 'on' if self.config.send_email else 'off'}")
        print(SEP)

    def _print_summary(self, result: PipelineResult):
        print(f"\n{SEP}")
        print("  ✅  Pipeline complete")
        print(f"  {result.summary()}")
        print(f"\n  Agent statuses:")
        for agent, status in result.agent_statuses.items():
            icon = "✅" if status.startswith("ok") else "⏭️ " if status.startswith("skip") else "❌"
            print(f"    {icon}  {agent:<14} {status}")
        if result.report_path:
            print(f"\n  Report saved → {result.report_path}")
        print(SEP + "\n")