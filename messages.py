"""
messages.py — Typed message contracts for inter-agent communication.

Every agent in the pipeline speaks through these dataclasses.
No agent imports another agent directly — only these message types.

Flow:
    HunterAgent      → AnomalyReport
    CorrelationAgent → EnrichedReport   (wraps AnomalyReport)
    RCAAgent         → InsightBundle    (wraps EnrichedReport)
    EmailAgent       → DeliveryReceipt  (wraps InsightBundle list)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums — shared vocabulary across all agents
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"   # |z| >= 3.0
    WARNING  = "warning"    # |z| >= 2.0
    INFO     = "info"       # |z| >= 1.5 (co-mover threshold)


class RiskDirection(str, Enum):
    DETERIORATING = "deteriorating"
    IMPROVING     = "improving"
    MIXED         = "mixed"
    ISOLATED      = "isolated"
    UNKNOWN       = "unknown"


class DeliveryStatus(str, Enum):
    SENT    = "sent"
    FAILED  = "failed"
    SKIPPED = "skipped"   # below threshold, no email warranted


# ---------------------------------------------------------------------------
# Tier 1 — HunterAgent output
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """
    Output of HunterAgent.run() for a single dimension.
    Represents one flagged anomaly segment.
    """
    dimension:     str                  # e.g. "regional"
    description:   str                  # human-readable dimension description
    segment:       dict                 # e.g. {"addr_state": "NV"}
    metric_name:   str                  # e.g. "avg_dti"
    metric_value:  float                # actual value for this segment
    metric_mean:   float                # population mean across all segments
    z_score:       float                # signed z-score
    volume:        int                  # number of loans in this segment
    anomaly_count: int                  # total flagged segments in this dimension
    severity:      Severity             # derived from |z_score|
    raw_df:        list[dict]           # full anomaly DataFrame as records
    scanned_at:    datetime = field(default_factory=datetime.now)

    @classmethod
    def from_insight(cls, insight: dict) -> "AnomalyReport":
        """
        Constructs from the dict format returned by RCAAgent.analyze_all()
        for backward compatibility during transition.
        """
        z = abs(insight.get("top_z", 0))
        severity = (
            Severity.CRITICAL if z >= 3.0 else
            Severity.WARNING  if z >= 2.0 else
            Severity.INFO
        )
        raw = insight.get("raw", [])
        top = raw[0] if raw else {}
        return cls(
            dimension     = insight["dimension"],
            description   = insight.get("description", ""),
            segment       = {k: v for k, v in top.items()
                             if k not in ("metric_value","avg_val","std_val","z_score","volume")},
            metric_name   = insight.get("metric_name", ""),
            metric_value  = top.get("metric_value", 0.0),
            metric_mean   = top.get("avg_val", 0.0),
            z_score       = top.get("z_score", 0.0),
            volume        = top.get("volume", 0),
            anomaly_count = insight.get("anomaly_count", 0),
            severity      = severity,
            raw_df        = raw,
        )


# ---------------------------------------------------------------------------
# Tier 2 — CorrelationAgent output
# ---------------------------------------------------------------------------

@dataclass
class CoMover:
    """A single co-moving metric for an anomalous segment."""
    metric_key:  str
    label:       str
    value:       float
    avg_val:     float
    z_score:     float
    signal:      str    # e.g. "⬆ elevated", "⬇ suppressed"
    direction:   str    # "higher_is_worse" | "lower_is_worse" | "neutral"


@dataclass
class EnrichedReport:
    """
    Output of CorrelationAgent.run().
    Wraps AnomalyReport and adds co-movement context.
    """
    anomaly:        AnomalyReport
    co_movers:      list[CoMover]
    risk_direction: RiskDirection
    narrative_hint: str             # one-line causal summary
    enriched_at:    datetime = field(default_factory=datetime.now)

    @property
    def dimension(self) -> str:
        return self.anomaly.dimension

    @property
    def severity(self) -> Severity:
        return self.anomaly.severity

    @property
    def top_co_movers(self) -> list[CoMover]:
        """Returns top 3 co-movers sorted by |z_score|."""
        return sorted(self.co_movers, key=lambda m: abs(m.z_score), reverse=True)[:3]

    @classmethod
    def from_raw(
        cls,
        anomaly: AnomalyReport,
        raw_enrichment: dict,
    ) -> "EnrichedReport":
        """Constructs from CorrelationAgent's raw dict output."""
        co_movers = [
            CoMover(
                metric_key = m["metric_key"],
                label      = m["label"],
                value      = m["value"],
                avg_val    = m["avg_val"],
                z_score    = m["z_score"],
                signal     = m["signal"],
                direction  = m["direction"],
            )
            for m in raw_enrichment.get("co_movers", [])
        ]
        direction_str = raw_enrichment.get("risk_direction", "unknown")
        try:
            risk_direction = RiskDirection(direction_str)
        except ValueError:
            risk_direction = RiskDirection.UNKNOWN

        return cls(
            anomaly        = anomaly,
            co_movers      = co_movers,
            risk_direction = risk_direction,
            narrative_hint = raw_enrichment.get("narrative_hint", ""),
        )


# ---------------------------------------------------------------------------
# Tier 3 — RCAAgent output
# ---------------------------------------------------------------------------

@dataclass
class InsightBundle:
    """
    Output of RCAAgent.run().
    Wraps EnrichedReport and adds the final narrative.
    """
    enriched:      EnrichedReport
    narrative:     str              # template or LLM-generated insight
    llm_generated: bool = False     # True if LLM was used
    generated_at:  datetime = field(default_factory=datetime.now)

    @property
    def dimension(self) -> str:
        return self.enriched.dimension

    @property
    def severity(self) -> Severity:
        return self.enriched.severity

    @property
    def risk_direction(self) -> RiskDirection:
        return self.enriched.risk_direction

    @property
    def top_z(self) -> float:
        return abs(self.enriched.anomaly.z_score)

    @property
    def segment_label(self) -> str:
        """Human-readable segment string e.g. 'addr_state=NV'."""
        return ", ".join(
            f"{k}={v}" for k, v in self.enriched.anomaly.segment.items()
        )


# ---------------------------------------------------------------------------
# Tier 4 — EmailAgent output
# ---------------------------------------------------------------------------

@dataclass
class RecipientGroup:
    """Defines an email recipient group and its filtering rules."""
    name:            str
    addresses:       list[str]
    min_severity:    Severity       = Severity.WARNING
    max_insights:    int            = 10    # max anomalies to include
    include_raw:     bool           = False # include raw data tables
    executive_mode:  bool           = False # shorter, boardroom-style copy


@dataclass
class DeliveryReceipt:
    """
    Output of EmailAgent.run().
    Records what was sent, to whom, and whether it succeeded.
    """
    status:       DeliveryStatus
    recipient:    str
    subject:      str
    insights_sent: int
    sent_at:      datetime = field(default_factory=datetime.now)
    error:        Optional[str] = None

    def __str__(self) -> str:
        if self.status == DeliveryStatus.SENT:
            return (
                f"✅ Sent to {self.recipient} — "
                f"{self.insights_sent} insights — '{self.subject}'"
            )
        elif self.status == DeliveryStatus.SKIPPED:
            return f"⏭️  Skipped {self.recipient} — below threshold"
        else:
            return f"❌ Failed to {self.recipient}: {self.error}"


# ---------------------------------------------------------------------------
# Pipeline result — top-level summary returned by Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Returned by Orchestrator.run().
    Contains the full audit trail across all agents and all dimensions.
    """
    run_id:          str
    started_at:      datetime
    completed_at:    datetime
    db_path:         str
    bundles:         list[InsightBundle]
    receipts:        list[DeliveryReceipt]
    agent_statuses:  dict[str, str]     # agent_name → "ok" | "error: ..."
    report_path:     Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def critical_count(self) -> int:
        return sum(1 for b in self.bundles if b.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for b in self.bundles if b.severity == Severity.WARNING)

    def summary(self) -> str:
        emails_sent = sum(
            1 for r in self.receipts if r.status == DeliveryStatus.SENT
        )
        return (
            f"Run {self.run_id} | "
            f"{len(self.bundles)} insights "
            f"({self.critical_count} critical, {self.warning_count} warnings) | "
            f"{emails_sent} emails sent | "
            f"{self.duration_seconds:.1f}s"
        )
