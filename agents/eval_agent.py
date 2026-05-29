"""
eval_agent.py — Narrative quality gate between RCAAgent and EmailAgent.

Scores each LLM-generated narrative on 4 dimensions using a second
(smaller, cheaper) LLM call. Falls back to the template narrative if
the score is below threshold — ensures no hallucinated or low-quality
insight ever reaches a portfolio manager.

Scoring dimensions:
    1. factual_grounding  — every claim traceable to the data (1-5)
    2. causal_validity    — hypothesis supported by co-movers (1-5)
    3. consistency        — stable, non-contradictory reasoning (1-5)
    4. tone               — appropriate for risk/compliance audience (1-5)

Decision logic:
    avg_score >= pass_threshold  → keep LLM narrative  ✅
    avg_score <  pass_threshold  → revert to template  ⚠️
    eval call fails              → keep LLM narrative with warning flag

Cost per eval call: ~200 input + 80 output tokens ≈ $0.001
"""

import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .messages import InsightBundle, Severity
from .rca_agent import NARRATIVE_TEMPLATES, _col, _fmt
from .hunter import DIMENSION_REGISTRY


# ---------------------------------------------------------------------------
# Eval result message
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Scores and decision for one InsightBundle narrative."""
    bundle_dimension:   str
    segment_label:      str
    factual_grounding:  float           # 1-5
    causal_validity:    float           # 1-5
    consistency:        float           # 1-5
    tone:               float           # 1-5
    avg_score:          float           # mean of above
    passed:             bool            # True = use LLM narrative
    flag:               str             # evaluator's one-line note
    narrative_used:     str             # "llm" | "template"
    eval_failed:        bool = False    # True if eval call itself errored
    evaluated_at:       datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        icon   = "✅" if self.passed else "⚠️ "
        source = "LLM" if self.narrative_used == "llm" else "template fallback"
        return (
            f"{icon} [{self.bundle_dimension}] "
            f"score={self.avg_score:.1f}/5 → {source} | {self.flag}"
        )


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = textwrap.dedent("""
    You are a strict narrative quality evaluator for a financial risk system.
    You will receive:
      - The raw anomaly data (numbers, z-scores, co-movers)
      - A generated narrative that a portfolio manager will read

    Score the narrative on exactly these 4 dimensions (each 1-5):

    factual_grounding (1-5):
        5 = every number/claim directly traceable to the data
        3 = mostly grounded, one unsupported claim
        1 = multiple claims not supported by data, or numbers are wrong

    causal_validity (1-5):
        5 = causal hypothesis logically follows from co-movers
        3 = plausible hypothesis but co-movers only partially support it
        1 = hypothesis contradicts co-mover evidence, or no hypothesis given

    consistency (1-5):
        5 = narrative is internally coherent, no contradictions
        3 = minor inconsistency that doesn't affect the conclusion
        1 = contradicts itself or draws opposite conclusions from same data

    tone (1-5):
        5 = precise, professional, suitable for a risk committee
        3 = acceptable but informal or imprecise in places
        1 = alarming without basis, casual, or inappropriate for audience

    Respond ONLY with valid JSON — no preamble, no markdown, no explanation:
    {
      "factual_grounding": <1-5>,
      "causal_validity": <1-5>,
      "consistency": <1-5>,
      "tone": <1-5>,
      "flag": "<one sentence: key strength or concern>"
    }
""").strip()


def build_eval_prompt(bundle: InsightBundle) -> str:
    """Builds the user-turn prompt for evaluating one InsightBundle narrative."""
    anomaly   = bundle.enriched.anomaly
    co_movers = bundle.enriched.top_co_movers

    co_str = "\n".join(
        f"  - {m.label}: value={round(m.value, 3)}, z={m.z_score:+.2f}, signal={m.signal}"
        for m in co_movers
    ) or "  (none)"

    return textwrap.dedent(f"""
        ANOMALY DATA:
          Dimension    : {anomaly.dimension}
          Segment      : {bundle.segment_label}
          Metric       : {anomaly.metric_name}
          Value        : {round(anomaly.metric_value, 3)}
          Portfolio avg: {round(anomaly.metric_mean, 3)}
          Z-score      : {anomaly.z_score:+.2f}
          Volume       : {anomaly.volume:,} loans
          Risk pattern : {bundle.risk_direction.value}

        CO-MOVING METRICS:
        {co_str}

        NARRATIVE TO EVALUATE:
        {bundle.narrative}
    """).strip()


# ---------------------------------------------------------------------------
# EvaluationAgent
# ---------------------------------------------------------------------------

class EvaluationAgent:
    """
    Quality gate between RCAAgent and EmailAgent.

    For LLM-generated narratives: scores each on 4 dimensions,
    falls back to template if score < pass_threshold.

    For template narratives: passes through immediately (no eval needed —
    templates are deterministic and already validated by design).

    Parameters
    ----------
    pass_threshold   : Minimum avg score (1-5) to keep LLM narrative.
                       Default 3.5 — requires at least "good" across all dims.
    api_key          : Anthropic API key. Reads ANTHROPIC_API_KEY env var if None.
    model            : Model for evaluation calls. Use a fast/cheap model.
    fail_open        : If True (default), keeps LLM narrative when eval call
                       itself fails. If False, falls back to template on error.
    """

    def __init__(
        self,
        pass_threshold: float = 3.5,
        api_key:        Optional[str] = None,
        model:          str = "claude-haiku-4-5-20251001",  # cheapest model for eval
        fail_open:      bool = True,
    ):
        self.pass_threshold = pass_threshold
        self.model          = model
        self.fail_open      = fail_open
        self._client        = None
        self._api_key       = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Client init (lazy)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _score_narrative(self, bundle: InsightBundle) -> EvalResult:
        """
        Calls the LLM evaluator and returns an EvalResult.
        Never raises — all errors produce an EvalResult with eval_failed=True.
        """
        prompt = build_eval_prompt(bundle)

        try:
            client   = self._get_client()
            response = client.messages.create(
                model      = self.model,
                max_tokens = 200,
                system     = EVAL_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw.strip())

            fg  = float(scores.get("factual_grounding", 3))
            cv  = float(scores.get("causal_validity",   3))
            con = float(scores.get("consistency",       3))
            tone= float(scores.get("tone",              3))
            avg = round((fg + cv + con + tone) / 4, 2)
            passed = avg >= self.pass_threshold

            return EvalResult(
                bundle_dimension  = bundle.dimension,
                segment_label     = bundle.segment_label,
                factual_grounding = fg,
                causal_validity   = cv,
                consistency       = con,
                tone              = tone,
                avg_score         = avg,
                passed            = passed,
                flag              = scores.get("flag", ""),
                narrative_used    = "llm" if passed else "template",
                eval_failed       = False,
            )

        except Exception as e:
            # Eval call failed — apply fail_open policy
            return EvalResult(
                bundle_dimension  = bundle.dimension,
                segment_label     = bundle.segment_label,
                factual_grounding = 0,
                causal_validity   = 0,
                consistency       = 0,
                tone              = 0,
                avg_score         = 0,
                passed            = self.fail_open,
                flag              = f"eval error: {str(e)[:80]}",
                narrative_used    = "llm" if self.fail_open else "template",
                eval_failed       = True,
            )

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _get_template_narrative(bundle: InsightBundle) -> str:
        """Regenerates the deterministic template narrative for a bundle."""
        import pandas as pd
        dim_key  = bundle.dimension
        spec     = DIMENSION_REGISTRY.get(dim_key, {})
        anomaly  = bundle.enriched.anomaly

        # Reconstruct a minimal DataFrame row for the template lambda
        row_data = {**anomaly.segment}
        row_data.update({
            "metric_value": anomaly.metric_value,
            "avg_val":      anomaly.metric_mean,
            "z_score":      anomaly.z_score,
            "volume":       anomaly.volume,
        })
        row = pd.Series(row_data)
        row.index = row.index.str.lower().str.strip() if hasattr(row.index, 'str') else row.index

        template_fn = NARRATIVE_TEMPLATES.get(dim_key)
        if template_fn:
            try:
                return template_fn(row, pd.DataFrame([row_data]), spec)
            except Exception:
                pass

        return (
            f"🚨 ANOMALY [{dim_key.upper()}]: segment {bundle.segment_label} "
            f"has metric value {round(anomaly.metric_value, 2)} "
            f"(mean: {round(anomaly.metric_mean, 2)}), "
            f"Z-score: {anomaly.z_score:+.2f}."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        bundle: InsightBundle,
    ) -> tuple[InsightBundle, EvalResult]:
        """
        Evaluates one InsightBundle.

        If the bundle used a template narrative (llm_generated=False),
        passes through immediately — no eval call made.

        If LLM-generated, scores it and potentially replaces the narrative
        with the template fallback.

        Returns (possibly modified InsightBundle, EvalResult).
        """
        # Template narratives pass through — no eval needed
        if not bundle.llm_generated:
            result = EvalResult(
                bundle_dimension  = bundle.dimension,
                segment_label     = bundle.segment_label,
                factual_grounding = 5.0,
                causal_validity   = 5.0,
                consistency       = 5.0,
                tone              = 5.0,
                avg_score         = 5.0,
                passed            = True,
                flag              = "template narrative — eval skipped",
                narrative_used    = "template",
            )
            return bundle, result

        # Score the LLM narrative
        eval_result = self._score_narrative(bundle)

        # If it failed the gate, swap in the template
        if not eval_result.passed:
            fallback_narrative = self._get_template_narrative(bundle)
            # Return a new InsightBundle with the template narrative
            from dataclasses import replace
            bundle = InsightBundle(
                enriched      = bundle.enriched,
                narrative     = fallback_narrative,
                llm_generated = False,   # mark as template after fallback
                generated_at  = bundle.generated_at,
            )

        return bundle, eval_result

    def evaluate_all(
        self,
        bundles: list[InsightBundle],
    ) -> tuple[list[InsightBundle], list[EvalResult]]:
        """
        Evaluates all bundles. Returns (evaluated bundles, eval results).
        Bundles with failed narratives are replaced with template fallbacks.
        """
        print("  🔬  Evaluation Agent: scoring narratives...")
        evaluated_bundles = []
        eval_results      = []

        for bundle in bundles:
            evaled_bundle, result = self.evaluate(bundle)
            evaluated_bundles.append(evaled_bundle)
            eval_results.append(result)
            print(f"     {result.summary()}")

        passed   = sum(1 for r in eval_results if r.passed and not r.eval_failed)
        fallback = sum(1 for r in eval_results if not r.passed)
        skipped  = sum(1 for r in eval_results if r.narrative_used == "template"
                       and r.flag == "template narrative — eval skipped")

        print(
            f"  ↳  {passed} passed | "
            f"{fallback} fell back to template | "
            f"{skipped} skipped (template mode)"
        )

        return evaluated_bundles, eval_results
