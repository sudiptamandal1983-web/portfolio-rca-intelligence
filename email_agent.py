"""
email_agent.py — Email delivery agent.

Receives a list of InsightBundles from RCAAgent, renders role-specific
HTML digests, and delivers via Gmail SMTP.

Recipient routing:
    risk_team        → full anomaly detail, all co-movers, raw tables
    portfolio_mgr    → executive summary, top insights only, clean prose
    custom groups    → configured via RecipientGroup dataclass

Gmail setup (one-time):
    1. Enable 2-Step Verification on your Google account
    2. Go to myaccount.google.com → Security → App Passwords
    3. Generate a password for "Mail" → copy the 16-char password
    4. Set env vars:
         export GMAIL_USER="you@gmail.com"
         export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
"""

import os
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from .messages import (
    InsightBundle, RecipientGroup, DeliveryReceipt,
    DeliveryStatus, Severity, RiskDirection,
)


# ---------------------------------------------------------------------------
# Default recipient groups — override via Orchestrator config
# ---------------------------------------------------------------------------

DEFAULT_RECIPIENT_GROUPS = [
    RecipientGroup(
        name           = "risk_team",
        addresses      = [],          # set via RISK_TEAM_EMAILS env var
        min_severity   = Severity.WARNING,
        max_insights   = 20,
        include_raw    = True,
        executive_mode = False,
    ),
    RecipientGroup(
        name           = "portfolio_mgr",
        addresses      = [],          # set via PORTFOLIO_MGR_EMAILS env var
        min_severity   = Severity.CRITICAL,
        max_insights   = 5,
        include_raw    = False,
        executive_mode = True,
    ),
]

# Severity display config
SEVERITY_CONFIG = {
    Severity.CRITICAL: {"emoji": "🔴", "label": "Critical",  "color": "#A32D2D", "bg": "#FCEBEB"},
    Severity.WARNING:  {"emoji": "🟡", "label": "Warning",   "color": "#854F0B", "bg": "#FAEEDA"},
    Severity.INFO:     {"emoji": "🔵", "label": "Info",      "color": "#185FA5", "bg": "#E6F1FB"},
}

RISK_DIRECTION_LABELS = {
    RiskDirection.DETERIORATING: "📉 Deteriorating",
    RiskDirection.IMPROVING:     "📈 Improving",
    RiskDirection.MIXED:         "↔️ Mixed signals",
    RiskDirection.ISOLATED:      "🔍 Isolated anomaly",
    RiskDirection.UNKNOWN:       "❓ Unknown",
}


class EmailAgent:
    """
    Renders and delivers portfolio insight digests via Gmail SMTP.

    Parameters
    ----------
    gmail_user        : Gmail address (reads GMAIL_USER env var if not passed).
    gmail_app_password: Gmail App Password (reads GMAIL_APP_PASSWORD env var).
    recipient_groups  : List of RecipientGroup configs. If None, reads from env.
    dry_run           : If True, renders emails but does not send — prints to terminal.
    """

    def __init__(
        self,
        gmail_user:         Optional[str] = None,
        gmail_app_password: Optional[str] = None,
        recipient_groups:   Optional[list[RecipientGroup]] = None,
        dry_run:            bool = False,
    ):
        self.gmail_user         = gmail_user or os.getenv("GMAIL_USER", "")
        self.gmail_app_password = gmail_app_password or os.getenv("GMAIL_APP_PASSWORD", "")
        self.dry_run            = dry_run
        self.recipient_groups   = recipient_groups or self._load_groups_from_env()

        if not self.dry_run:
            self._validate_credentials()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_groups_from_env(self) -> list[RecipientGroup]:
        """
        Reads recipient email addresses from env vars.
        RISK_TEAM_EMAILS and PORTFOLIO_MGR_EMAILS are comma-separated.

        Example:
            export RISK_TEAM_EMAILS="alice@bank.com,bob@bank.com"
            export PORTFOLIO_MGR_EMAILS="cfo@bank.com"
        """
        groups = []
        for group in DEFAULT_RECIPIENT_GROUPS:
            env_key = group.name.upper() + "_EMAILS"
            raw = os.getenv(env_key, "")
            addresses = [e.strip() for e in raw.split(",") if e.strip()]
            if addresses:
                groups.append(RecipientGroup(
                    name           = group.name,
                    addresses      = addresses,
                    min_severity   = group.min_severity,
                    max_insights   = group.max_insights,
                    include_raw    = group.include_raw,
                    executive_mode = group.executive_mode,
                ))
        return groups

    def _validate_credentials(self):
        if not self.gmail_user:
            raise ValueError(
                "Gmail user not set. "
                "Pass gmail_user= or set GMAIL_USER env var."
            )
        if not self.gmail_app_password:
            raise ValueError(
                "Gmail App Password not set. "
                "Pass gmail_app_password= or set GMAIL_APP_PASSWORD env var.\n"
                "Setup: myaccount.google.com → Security → App Passwords"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, bundles: list[InsightBundle]) -> list[DeliveryReceipt]:
        """
        Main entry point. Routes bundles to recipient groups,
        renders digests, and sends emails.

        Returns a list of DeliveryReceipt — one per recipient address.
        """
        receipts = []

        if not bundles:
            print("  📭  Email agent: no insights to deliver.")
            return receipts

        if not self.recipient_groups:
            print(
                "  ⚠️  Email agent: no recipient groups configured.\n"
                "      Set RISK_TEAM_EMAILS or PORTFOLIO_MGR_EMAILS env vars."
            )
            return receipts

        print(f"  📧  Email agent: routing {len(bundles)} insights to "
              f"{len(self.recipient_groups)} recipient group(s)...")

        for group in self.recipient_groups:
            # Filter bundles by severity threshold for this group
            filtered = [
                b for b in bundles
                if self._severity_rank(b.severity) >=
                   self._severity_rank(group.min_severity)
            ][:group.max_insights]

            if not filtered:
                for addr in group.addresses:
                    receipts.append(DeliveryReceipt(
                        status        = DeliveryStatus.SKIPPED,
                        recipient     = addr,
                        subject       = "",
                        insights_sent = 0,
                    ))
                continue

            subject = self._build_subject(filtered, group)
            html    = self._render_digest(filtered, group)

            for addr in group.addresses:
                receipt = self._send(addr, subject, html, len(filtered))
                receipts.append(receipt)
                print(f"     {receipt}")

        return receipts

    # ------------------------------------------------------------------
    # Email rendering
    # ------------------------------------------------------------------

    def _build_subject(
        self, bundles: list[InsightBundle], group: RecipientGroup
    ) -> str:
        critical = sum(1 for b in bundles if b.severity == Severity.CRITICAL)
        warning  = sum(1 for b in bundles if b.severity == Severity.WARNING)
        date_str = datetime.now().strftime("%d %b %Y")

        prefix = "🔴 CRITICAL" if critical else "🟡 WARNING"
        return (
            f"{prefix} | Portfolio RCA Digest — {date_str} | "
            f"{critical} critical, {warning} warnings"
        )

    def _render_digest(
        self, bundles: list[InsightBundle], group: RecipientGroup
    ) -> str:
        """Renders a full HTML email digest."""

        # Sort by severity then z-score
        sorted_bundles = sorted(
            bundles,
            key=lambda b: (self._severity_rank(b.severity), b.top_z),
            reverse=True,
        )

        insight_sections = "\n".join(
            self._render_insight_card(b, group) for b in sorted_bundles
        )

        critical = sum(1 for b in bundles if b.severity == Severity.CRITICAL)
        warning  = sum(1 for b in bundles if b.severity == Severity.WARNING)
        date_str = datetime.now().strftime("%A, %d %B %Y %H:%M")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f0; margin: 0; padding: 24px; color: #2c2c2a; }}
  .container {{ max-width: 680px; margin: 0 auto; }}
  .header {{ background: #2c2c2a; color: #f1efe8; padding: 24px 28px;
             border-radius: 12px 12px 0 0; }}
  .header h1 {{ margin: 0 0 4px; font-size: 20px; font-weight: 500; }}
  .header p  {{ margin: 0; font-size: 13px; opacity: 0.65; }}
  .summary {{ background: #ffffff; padding: 16px 28px;
              border-left: 4px solid #2c2c2a; margin-bottom: 16px;
              display: flex; gap: 24px; }}
  .stat {{ text-align: center; }}
  .stat .n {{ font-size: 28px; font-weight: 500; line-height: 1; }}
  .stat .l {{ font-size: 12px; color: #888780; margin-top: 2px; }}
  .card {{ background: #ffffff; border-radius: 10px; margin-bottom: 14px;
           overflow: hidden; border: 1px solid #e8e6df; }}
  .card-header {{ padding: 14px 20px; display: flex;
                  align-items: center; gap: 10px; }}
  .badge {{ font-size: 11px; font-weight: 500; padding: 3px 8px;
            border-radius: 4px; }}
  .card-body {{ padding: 14px 20px; }}
  .narrative {{ font-size: 14px; line-height: 1.65; color: #3d3d3a;
                margin: 0 0 14px; }}
  .metrics-row {{ display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }}
  .metric {{ background: #f5f5f0; border-radius: 6px; padding: 8px 12px; flex: 1;
             min-width: 120px; }}
  .metric .mv {{ font-size: 18px; font-weight: 500; }}
  .metric .ml {{ font-size: 11px; color: #888780; margin-top: 2px; }}
  .co-movers {{ border-top: 1px solid #e8e6df; padding-top: 12px; }}
  .co-movers h4 {{ font-size: 12px; font-weight: 500; color: #888780;
                   margin: 0 0 8px; text-transform: uppercase; letter-spacing: .04em; }}
  .co-mover-row {{ display: flex; justify-content: space-between;
                   align-items: center; padding: 5px 0;
                   border-bottom: 1px solid #f1efe8; font-size: 13px; }}
  .co-mover-row:last-child {{ border-bottom: none; }}
  .signal {{ font-size: 12px; }}
  .raw-table {{ width: 100%; border-collapse: collapse; font-size: 12px;
                margin-top: 12px; }}
  .raw-table th {{ background: #f1efe8; padding: 6px 10px; text-align: left;
                   font-weight: 500; color: #5f5e5a; }}
  .raw-table td {{ padding: 5px 10px; border-bottom: 1px solid #f1efe8; }}
  .footer {{ text-align: center; font-size: 11px; color: #b4b2a9;
             padding: 20px 0; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Portfolio RCA Digest</h1>
    <p>{date_str} &nbsp;·&nbsp; Recipient group: {group.name}</p>
  </div>

  <div class="summary">
    <div class="stat">
      <div class="n" style="color:#A32D2D">{critical}</div>
      <div class="l">Critical</div>
    </div>
    <div class="stat">
      <div class="n" style="color:#854F0B">{warning}</div>
      <div class="l">Warnings</div>
    </div>
    <div class="stat">
      <div class="n">{len(bundles)}</div>
      <div class="l">Total insights</div>
    </div>
    <div class="stat">
      <div class="n">{len(set(b.dimension for b in bundles))}</div>
      <div class="l">Dimensions flagged</div>
    </div>
  </div>

  {insight_sections}

  <div class="footer">
    Generated by Banking Portfolio RCA Pipeline &nbsp;·&nbsp;
    Auto-generated — do not reply
  </div>

</div>
</body>
</html>"""

    def _render_insight_card(
        self, bundle: InsightBundle, group: RecipientGroup
    ) -> str:
        sev     = SEVERITY_CONFIG[bundle.severity]
        anomaly = bundle.enriched.anomaly
        seg_str = bundle.segment_label
        direction_label = RISK_DIRECTION_LABELS.get(bundle.risk_direction, "")

        # Metrics row
        delta_pct = (
            ((anomaly.metric_value - anomaly.metric_mean) / anomaly.metric_mean * 100)
            if anomaly.metric_mean else 0
        )
        delta_sign = "+" if delta_pct > 0 else ""

        metrics_html = f"""
        <div class="metrics-row">
          <div class="metric">
            <div class="mv">{round(anomaly.metric_value, 2)}</div>
            <div class="ml">Segment value</div>
          </div>
          <div class="metric">
            <div class="mv">{round(anomaly.metric_mean, 2)}</div>
            <div class="ml">Portfolio mean</div>
          </div>
          <div class="metric">
            <div class="mv">{delta_sign}{round(delta_pct, 1)}%</div>
            <div class="ml">Deviation</div>
          </div>
          <div class="metric">
            <div class="mv">{bundle.top_z:+.2f}σ</div>
            <div class="ml">Z-score</div>
          </div>
          <div class="metric">
            <div class="mv">{anomaly.volume:,}</div>
            <div class="ml">Loans</div>
          </div>
        </div>"""

        # Co-movers table
        co_movers_html = ""
        if bundle.enriched.co_movers and not group.executive_mode:
            rows = "\n".join(
                f"""<div class="co-mover-row">
                      <span>{m.label}</span>
                      <span class="signal">{m.signal}</span>
                      <span style="color:#888780">{round(m.value, 2)} &nbsp; z={m.z_score:+.2f}</span>
                    </div>"""
                for m in bundle.enriched.top_co_movers
            )
            co_movers_html = f"""
            <div class="co-movers">
              <h4>Co-moving metrics</h4>
              {rows}
              <div style="font-size:11px;color:#b4b2a9;margin-top:6px">
                Overall risk pattern: {direction_label}
              </div>
            </div>"""

        # Raw data table (risk team only)
        raw_html = ""
        if group.include_raw and anomaly.raw_df:
            headers = list(anomaly.raw_df[0].keys())
            header_row = "".join(f"<th>{h}</th>" for h in headers)
            data_rows  = "\n".join(
                "<tr>" + "".join(
                    f"<td>{round(v, 3) if isinstance(v, float) else v}</td>"
                    for v in row.values()
                ) + "</tr>"
                for row in anomaly.raw_df[:5]
            )
            raw_html = f"""
            <details style="margin-top:12px">
              <summary style="font-size:12px;color:#888780;cursor:pointer">
                Raw anomaly data ({len(anomaly.raw_df)} rows)
              </summary>
              <table class="raw-table">
                <thead><tr>{header_row}</tr></thead>
                <tbody>{data_rows}</tbody>
              </table>
            </details>"""

        return f"""
<div class="card">
  <div class="card-header" style="background:{sev['bg']}">
    <span style="font-size:16px">{sev['emoji']}</span>
    <div style="flex:1">
      <strong style="font-size:14px;color:{sev['color']}">{bundle.dimension.upper().replace('_',' ')}</strong>
      &nbsp;
      <span class="badge" style="background:{sev['color']}20;color:{sev['color']}">{sev['label']}</span>
    </div>
    <span style="font-size:12px;color:#888780">{seg_str}</span>
  </div>
  <div class="card-body">
    <p class="narrative">{bundle.narrative}</p>
    {metrics_html}
    {co_movers_html}
    {raw_html}
  </div>
</div>"""

    # ------------------------------------------------------------------
    # SMTP delivery
    # ------------------------------------------------------------------

    def _send(
        self,
        to_address:    str,
        subject:       str,
        html:          str,
        insights_sent: int,
    ) -> DeliveryReceipt:
        """Sends the HTML email via Gmail SMTP."""

        if self.dry_run:
            print(f"\n{'─'*60}")
            print(f"  DRY RUN — would send to: {to_address}")
            print(f"  Subject : {subject}")
            print(f"  Insights: {insights_sent}")
            print(f"{'─'*60}\n")
            return DeliveryReceipt(
                status        = DeliveryStatus.SENT,
                recipient     = to_address,
                subject       = subject,
                insights_sent = insights_sent,
            )

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.gmail_user
            msg["To"]      = to_address

            # Plain text fallback
            plain = f"Portfolio RCA Digest — {insights_sent} insights. View in an HTML-capable client."
            msg.attach(MIMEText(plain, "plain"))
            msg.attach(MIMEText(html,  "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.gmail_user, self.gmail_app_password)
                server.sendmail(self.gmail_user, to_address, msg.as_string())

            return DeliveryReceipt(
                status        = DeliveryStatus.SENT,
                recipient     = to_address,
                subject       = subject,
                insights_sent = insights_sent,
            )

        except smtplib.SMTPAuthenticationError:
            error = (
                "Gmail authentication failed. "
                "Check your App Password at myaccount.google.com → Security → App Passwords."
            )
            return DeliveryReceipt(
                status        = DeliveryStatus.FAILED,
                recipient     = to_address,
                subject       = subject,
                insights_sent = 0,
                error         = error,
            )
        except Exception as e:
            return DeliveryReceipt(
                status        = DeliveryStatus.FAILED,
                recipient     = to_address,
                subject       = subject,
                insights_sent = 0,
                error         = str(e),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_rank(severity: Severity) -> int:
        return {Severity.INFO: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}[severity]
