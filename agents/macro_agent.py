"""
macro_agent.py — FRED macro data enrichment agent.

Fetches 7 macro time series from the Federal Reserve (FRED API),
joins them to anomalous loan segments at two points:
  1. issue_d      — conditions at origination
  2. last_pymnt_d — conditions at last payment / performance

Produces MacroContext messages that flow into RCAAgent (narrative)
and the dashboard (macro charts).

FRED API key (free):
    1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
    2. Create a free account → request API key (instant)
    3. export FRED_API_KEY="your_key_here"

No API key? The agent falls back to cached/embedded data for
the 2010-2020 range that covers most Lending Club loans.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import pandas as pd
import duckdb


# ---------------------------------------------------------------------------
# FRED series registry
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "fed_funds_rate": {
        "series_id":   "FEDFUNDS",
        "label":       "Fed Funds Rate (%)",
        "description": "Federal funds effective rate — primary monetary policy lever",
        "frequency":   "monthly",
        "direction":   "higher_tightens_credit",
    },
    "unemployment": {
        "series_id":   "UNRATE",
        "label":       "Unemployment Rate (%)",
        "description": "US civilian unemployment rate — borrower stress indicator",
        "frequency":   "monthly",
        "direction":   "higher_is_worse",
    },
    "cpi": {
        "series_id":   "CPIAUCSL",
        "label":       "CPI (Index)",
        "description": "Consumer Price Index — inflation environment",
        "frequency":   "monthly",
        "direction":   "neutral",
    },
    "consumer_sentiment": {
        "series_id":   "UMCSENT",
        "label":       "Consumer Sentiment (Index)",
        "description": "University of Michigan consumer sentiment — behavioural signal",
        "frequency":   "monthly",
        "direction":   "lower_is_worse",
    },
    "hy_credit_spread": {
        "series_id":   "BAMLH0A0HYM2",
        "label":       "HY Credit Spread (%)",
        "description": "ICE BofA high yield spread — market risk appetite",
        "frequency":   "daily",
        "direction":   "higher_is_worse",
    },
    "yield_curve": {
        "series_id":   "T10Y2Y",
        "label":       "Yield Curve (10Y-2Y, %)",
        "description": "10Y minus 2Y treasury spread — recession signal when negative",
        "frequency":   "daily",
        "direction":   "negative_signals_recession",
    },
    "cc_delinquency": {
        "series_id":   "DRCCLACBS",
        "label":       "Credit Card Delinquency Rate (%)",
        "description": "Commercial bank credit card delinquency rate — consumer credit stress",
        "frequency":   "quarterly",
        "direction":   "higher_is_worse",
    },
}

# Cache file path
MACRO_CACHE_PATH = "data/macro_cache.json"


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class MacroSnapshot:
    """Macro conditions at a specific month."""
    month:              str         # YYYY-MM
    fed_funds_rate:     Optional[float] = None
    unemployment:       Optional[float] = None
    cpi:                Optional[float] = None
    consumer_sentiment: Optional[float] = None
    hy_credit_spread:   Optional[float] = None
    yield_curve:        Optional[float] = None
    cc_delinquency:     Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def narrative_context(self) -> str:
        """Returns a compact macro description for LLM prompts."""
        parts = []
        if self.fed_funds_rate is not None:
            parts.append(f"fed funds {self.fed_funds_rate:.2f}%")
        if self.unemployment is not None:
            parts.append(f"unemployment {self.unemployment:.1f}%")
        if self.cpi is not None:
            parts.append(f"CPI {self.cpi:.1f}")
        if self.consumer_sentiment is not None:
            parts.append(f"sentiment {self.consumer_sentiment:.1f}")
        if self.hy_credit_spread is not None:
            parts.append(f"HY spread {self.hy_credit_spread:.2f}%")
        if self.yield_curve is not None:
            sign = "+" if self.yield_curve >= 0 else ""
            parts.append(f"yield curve {sign}{self.yield_curve:.2f}%")
        if self.cc_delinquency is not None:
            parts.append(f"CC delinquency {self.cc_delinquency:.2f}%")
        return "; ".join(parts) if parts else "macro data unavailable"


@dataclass
class MacroContext:
    """
    Full macro enrichment for one anomalous segment.
    Contains snapshots at origination and performance date,
    plus a derived narrative hint for RCA.
    """
    dimension:          str
    segment:            dict
    origination_month:  Optional[str]           # avg issue_d for segment
    performance_month:  Optional[str]           # avg last_pymnt_d for segment
    at_origination:     Optional[MacroSnapshot] = None
    at_performance:     Optional[MacroSnapshot] = None
    macro_narrative:    str = ""                # one-line macro summary
    regime_label:       str = ""                # e.g. "tightening cycle", "stress period"
    fetched_at:         datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "dimension":         self.dimension,
            "segment":           self.segment,
            "origination_month": self.origination_month,
            "performance_month": self.performance_month,
            "at_origination":    self.at_origination.to_dict() if self.at_origination else {},
            "at_performance":    self.at_performance.to_dict() if self.at_performance else {},
            "macro_narrative":   self.macro_narrative,
            "regime_label":      self.regime_label,
        }


# ---------------------------------------------------------------------------
# FRED fetcher
# ---------------------------------------------------------------------------

class FREDFetcher:
    """
    Fetches time series from the FRED API and resamples to monthly frequency.
    Results are cached to data/macro_cache.json to avoid repeated API calls.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or os.getenv("FRED_API_KEY", "")
        self._cache   = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(MACRO_CACHE_PATH):
            try:
                with open(MACRO_CACHE_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(MACRO_CACHE_PATH), exist_ok=True)
        with open(MACRO_CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def fetch_series(
        self,
        series_id:  str,
        start_date: str = "2007-01-01",
        end_date:   str = "2020-12-31",
    ) -> pd.Series:
        """
        Fetches a FRED series and returns a monthly pd.Series indexed by YYYY-MM.
        Uses cache if available — only calls API on cache miss.
        """
        cache_key = f"{series_id}_{start_date}_{end_date}"

        if cache_key in self._cache:
            data = self._cache[cache_key]
            return pd.Series(data["values"], index=data["dates"], name=series_id)

        if not self.api_key:
            print(f"    ⚠️  No FRED API key — using fallback data for {series_id}")
            return self._fallback_series(series_id)

        try:
            params = urllib.parse.urlencode({
                "series_id":         series_id,
                "api_key":           self.api_key,
                "file_type":         "json",
                "observation_start": start_date,
                "observation_end":   end_date,
                "frequency":         "m",   # resample to monthly
                "aggregation_method":"avg",
            })
            url = f"{self.BASE_URL}?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = json.loads(resp.read())

            observations = raw.get("observations", [])
            values, dates = [], []
            for obs in observations:
                if obs["value"] != ".":
                    try:
                        dates.append(obs["date"][:7])   # YYYY-MM
                        values.append(float(obs["value"]))
                    except ValueError:
                        pass

            series = pd.Series(values, index=dates, name=series_id)

            # Cache it
            self._cache[cache_key] = {"dates": dates, "values": values}
            self._save_cache()

            time.sleep(0.2)  # be polite to FRED API
            return series

        except Exception as e:
            print(f"    ⚠️  FRED fetch failed for {series_id}: {e} — using fallback")
            return self._fallback_series(series_id)

    def _fallback_series(self, series_id: str) -> pd.Series:
        """
        Embedded fallback data covering 2010-2020 for the 7 key series.
        Used when no API key is set or API call fails.
        Values are approximate monthly averages sourced from public FRED data.
        """
        fallback = {
            "FEDFUNDS": {
                "2010-01": 0.11, "2011-01": 0.07, "2012-01": 0.07,
                "2013-01": 0.07, "2014-01": 0.07, "2014-11": 0.09,
                "2015-01": 0.11, "2015-10": 0.12, "2015-12": 0.24,
                "2016-01": 0.34, "2016-06": 0.38, "2016-12": 0.54,
                "2017-01": 0.66, "2017-06": 1.04, "2017-12": 1.30,
                "2018-01": 1.42, "2018-06": 1.82, "2018-12": 2.27,
                "2019-01": 2.40, "2019-06": 2.38, "2019-12": 1.55,
                "2020-01": 1.55, "2020-03": 0.65, "2020-06": 0.08,
            },
            "UNRATE": {
                "2010-01": 9.8,  "2011-01": 9.1,  "2012-01": 8.3,
                "2013-01": 8.0,  "2014-01": 6.6,  "2014-11": 5.8,
                "2015-01": 5.7,  "2015-10": 5.0,  "2016-01": 4.9,
                "2016-06": 4.9,  "2016-12": 4.7,  "2017-01": 4.7,
                "2017-06": 4.4,  "2017-12": 4.1,  "2018-01": 4.1,
                "2018-06": 4.0,  "2018-12": 3.9,  "2019-01": 4.0,
                "2019-06": 3.7,  "2019-12": 3.5,  "2020-01": 3.5,
                "2020-03": 4.4,  "2020-06": 11.1,
            },
            "CPIAUCSL": {
                "2010-01": 217.5, "2011-01": 220.2, "2012-01": 226.7,
                "2013-01": 230.3, "2014-01": 233.9, "2014-11": 237.3,
                "2015-01": 234.7, "2015-10": 237.8, "2016-01": 236.9,
                "2016-06": 240.2, "2016-12": 241.4, "2017-01": 242.8,
                "2017-06": 244.9, "2017-12": 246.5, "2018-01": 247.9,
                "2018-06": 251.6, "2018-12": 251.2, "2019-01": 251.7,
                "2019-06": 255.7, "2019-12": 256.6, "2020-01": 257.0,
            },
            "UMCSENT": {
                "2010-01": 74.4, "2011-01": 74.2, "2012-01": 75.0,
                "2013-01": 73.8, "2014-01": 81.2, "2014-11": 88.8,
                "2015-01": 98.1, "2015-10": 90.0, "2016-01": 92.0,
                "2016-06": 93.5, "2016-12": 98.2, "2017-01": 98.5,
                "2017-06": 95.1, "2017-12": 95.9, "2018-01": 95.7,
                "2018-06": 98.2, "2018-12": 98.3, "2019-01": 91.2,
                "2019-06": 98.2, "2019-12": 99.3, "2020-01": 99.8,
                "2020-03": 89.1, "2020-06": 78.1,
            },
            "BAMLH0A0HYM2": {
                "2010-01": 6.17, "2011-01": 5.23, "2012-01": 6.80,
                "2013-01": 5.26, "2014-01": 4.00, "2014-11": 4.72,
                "2015-01": 5.06, "2015-10": 6.35, "2016-01": 8.35,
                "2016-06": 6.86, "2016-12": 4.14, "2017-01": 3.98,
                "2017-06": 3.73, "2017-12": 3.41, "2018-01": 3.45,
                "2018-06": 3.42, "2018-12": 5.26, "2019-01": 5.00,
                "2019-06": 3.98, "2019-12": 3.31, "2020-01": 3.64,
                "2020-03": 8.81, "2020-06": 6.59,
            },
            "T10Y2Y": {
                "2010-01": 2.77, "2011-01": 2.65, "2012-01": 1.57,
                "2013-01": 1.59, "2014-01": 2.55, "2014-11": 1.85,
                "2015-01": 1.44, "2015-10": 1.44, "2016-01": 1.19,
                "2016-06": 0.92, "2016-12": 1.25, "2017-01": 1.27,
                "2017-06": 0.98, "2017-12": 0.51, "2018-01": 0.55,
                "2018-06": 0.37, "2018-12": 0.14, "2019-01": 0.17,
                "2019-06": -0.09,"2019-12": 0.34, "2020-01": 0.34,
                "2020-03": -0.49,"2020-06": 0.58,
            },
            "DRCCLACBS": {
                "2010-01": 6.61, "2011-01": 5.64, "2012-01": 4.44,
                "2013-01": 3.74, "2014-01": 3.05, "2014-10": 2.81,
                "2015-01": 2.82, "2015-10": 2.87, "2016-01": 2.93,
                "2016-07": 2.99, "2016-12": 3.09, "2017-01": 3.22,
                "2017-07": 3.40, "2017-12": 3.59, "2018-01": 3.63,
                "2018-07": 3.67, "2018-12": 3.74, "2019-01": 3.80,
                "2019-07": 3.77, "2019-12": 3.77, "2020-01": 3.82,
            },
        }
        data  = fallback.get(series_id, {})
        return pd.Series(data, name=series_id) if data else pd.Series(name=series_id)


# ---------------------------------------------------------------------------
# MacroAgent
# ---------------------------------------------------------------------------

class MacroAgent:
    """
    Enriches anomalous loan segments with macro context from FRED.

    For each anomaly segment:
    1. Queries DuckDB to get avg issue_d and avg last_pymnt_d for the segment
    2. Looks up all 7 macro series at both dates
    3. Derives regime label and narrative hint
    4. Returns MacroContext messages for RCA and dashboard

    Parameters
    ----------
    db_path   : Path to DuckDB database
    api_key   : FRED API key (reads FRED_API_KEY env var if None)
    start_date: Earliest date to fetch macro data for
    end_date  : Latest date to fetch macro data for
    """

    def __init__(
        self,
        db_path:    str = "data/portfolio.db",
        api_key:    Optional[str] = None,
        start_date: str = "2007-01-01",
        end_date:   str = "2020-12-31",
    ):
        self.db_path    = db_path
        self.start_date = start_date
        self.end_date   = end_date
        self.fetcher    = FREDFetcher(api_key)
        self._macro_df  = None   # loaded once, reused across all segments

    # ------------------------------------------------------------------
    # Load all macro series into one DataFrame
    # ------------------------------------------------------------------

    def _load_macro_panel(self) -> pd.DataFrame:
        """
        Fetches all 7 FRED series and merges into a single monthly DataFrame.
        Cached in self._macro_df after first call.
        """
        if self._macro_df is not None:
            return self._macro_df

        print("  🌐  Macro Agent: fetching FRED series...")
        series_dict = {}
        for name, spec in FRED_SERIES.items():
            print(f"     ↳ {spec['series_id']} — {spec['label']}")
            s = self.fetcher.fetch_series(
                spec["series_id"], self.start_date, self.end_date
            )
            series_dict[name] = s

        df = pd.DataFrame(series_dict)
        df.index.name = "month"
        df = df.sort_index()

        # Forward-fill sparse series (quarterly → monthly)
        df = df.ffill().bfill()
        self._macro_df = df
        print(f"  ✅  Macro panel ready: {len(df)} months × {len(df.columns)} series")
        return df

    # ------------------------------------------------------------------
    # Get segment dates from DuckDB
    # ------------------------------------------------------------------

    def _get_segment_dates(
        self,
        dimension_cols: list[str],
        segment:        dict,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Queries DuckDB for the avg issue_d and avg last_pymnt_d
        for a specific segment — returns (origination_month, performance_month).
        """
        where_parts = [
            f"{col} = '{val}'" for col, val in segment.items()
            if col in dimension_cols
        ]
        if not where_parts:
            return None, None

        where_clause = " AND ".join(where_parts)

        query = f"""
        SELECT
            STRFTIME('%Y-%m', MAKE_TIMESTAMP(CAST(AVG(EPOCH(STRPTIME(issue_d, '%b-%Y'))) AS BIGINT) * 1000000))
                AS avg_issue_month,
            STRFTIME('%Y-%m', MAKE_TIMESTAMP(CAST(AVG(EPOCH(STRPTIME(last_pymnt_d, '%b-%Y'))) AS BIGINT) * 1000000))
                AS avg_perf_month
        FROM loans
        WHERE {where_clause}
          AND issue_d IS NOT NULL
          AND issue_d != ''
        """

        try:
            with duckdb.connect(self.db_path) as con:
                result = con.execute(query).df()
            if result.empty:
                return None, None
            row = result.iloc[0]
            orig = str(row["avg_issue_month"])[:7] if row["avg_issue_month"] else None
            perf = str(row["avg_perf_month"])[:7]  if row["avg_perf_month"]  else None
            return orig, perf
        except Exception as e:
            print(f"    ⚠️  Date query failed for {segment}: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Look up macro snapshot for a given month
    # ------------------------------------------------------------------

    def _get_snapshot(
        self, month: Optional[str], macro_df: pd.DataFrame
    ) -> Optional[MacroSnapshot]:
        """
        Returns a MacroSnapshot for the given YYYY-MM month.
        Uses nearest available month if exact match not found.
        """
        if not month or macro_df.empty:
            return None

        # Find nearest month using string comparison
        if month in macro_df.index:
            row = macro_df.loc[month]
        else:
            # Find closest available by string sort distance
            available = sorted(macro_df.index.tolist())
            if not available:
                return None
            # Pick the closest month by finding where it would be inserted
            import bisect
            pos = bisect.bisect_left(available, month)
            if pos == 0:
                nearest = available[0]
            elif pos >= len(available):
                nearest = available[-1]
            else:
                # Compare adjacent months
                before = available[pos - 1]
                after  = available[pos]
                nearest = before if (month < after or pos == len(available)) else after
            row   = macro_df.loc[nearest]
            month = nearest

        def safe(col):
            v = row.get(col)
            return float(v) if v is not None and not pd.isna(v) else None

        return MacroSnapshot(
            month              = month,
            fed_funds_rate     = safe("fed_funds_rate"),
            unemployment       = safe("unemployment"),
            cpi                = safe("cpi"),
            consumer_sentiment = safe("consumer_sentiment"),
            hy_credit_spread   = safe("hy_credit_spread"),
            yield_curve        = safe("yield_curve"),
            cc_delinquency     = safe("cc_delinquency"),
        )

    # ------------------------------------------------------------------
    # Derive regime label from macro snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_regime(snap: Optional[MacroSnapshot]) -> str:
        """
        Classifies the macro environment into a human-readable regime label.
        Used in narratives and dashboard.
        """
        if not snap:
            return "unknown regime"

        fed   = snap.fed_funds_rate   or 0
        unemp = snap.unemployment     or 0
        curve = snap.yield_curve      or 0
        spread= snap.hy_credit_spread or 0
        sent  = snap.consumer_sentiment or 100

        if fed >= 2.0 and curve < 0.5:
            return "late-cycle tightening"
        elif fed >= 1.0 and curve > 1.0:
            return "normalisation cycle"
        elif fed < 0.25 and unemp > 7.0:
            return "crisis / zero-rate environment"
        elif fed < 0.25 and unemp < 5.0:
            return "accommodative / recovery"
        elif spread > 6.0:
            return "credit stress / risk-off"
        elif curve < 0:
            return "yield curve inversion / recession signal"
        elif sent < 80:
            return "consumer pessimism"
        elif unemp < 4.5 and spread < 4.0:
            return "benign credit environment"
        else:
            return "moderate macro environment"

    # ------------------------------------------------------------------
    # Build macro narrative hint
    # ------------------------------------------------------------------

    @staticmethod
    def _build_macro_narrative(
        orig_snap: Optional[MacroSnapshot],
        perf_snap: Optional[MacroSnapshot],
        regime:    str,
    ) -> str:
        """
        Builds a one-paragraph macro narrative comparing origination
        and performance environments.
        """
        if not orig_snap and not perf_snap:
            return "Macro context unavailable for this segment."

        parts = []

        if orig_snap:
            parts.append(
                f"At origination ({orig_snap.month}), the macro environment was "
                f"characterised by a {regime} — {orig_snap.narrative_context()}."
            )

        if perf_snap and orig_snap and perf_snap.month != orig_snap.month:
            # Compare rate changes
            rate_delta = None
            if orig_snap.fed_funds_rate and perf_snap.fed_funds_rate:
                rate_delta = perf_snap.fed_funds_rate - orig_snap.fed_funds_rate

            unemp_delta = None
            if orig_snap.unemployment and perf_snap.unemployment:
                unemp_delta = perf_snap.unemployment - orig_snap.unemployment

            changes = []
            if rate_delta is not None:
                direction = "rose" if rate_delta > 0 else "fell"
                changes.append(
                    f"fed funds {direction} {abs(rate_delta):.2f}pp "
                    f"to {perf_snap.fed_funds_rate:.2f}%"
                )
            if unemp_delta is not None:
                direction = "increased" if unemp_delta > 0 else "decreased"
                changes.append(
                    f"unemployment {direction} {abs(unemp_delta):.1f}pp "
                    f"to {perf_snap.unemployment:.1f}%"
                )

            if changes:
                parts.append(
                    f"Between origination and last payment ({perf_snap.month}), "
                    + " and ".join(changes) + "."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enrich_segment(
        self,
        dimension:      str,
        segment:        dict,
        dimension_cols: list[str],
    ) -> MacroContext:
        """Enriches a single anomaly segment with macro context."""
        macro_df = self._load_macro_panel()

        orig_month, perf_month = self._get_segment_dates(dimension_cols, segment)

        orig_snap = self._get_snapshot(orig_month, macro_df)
        perf_snap = self._get_snapshot(perf_month, macro_df)

        regime  = self._classify_regime(orig_snap or perf_snap)
        narrative = self._build_macro_narrative(orig_snap, perf_snap, regime)

        return MacroContext(
            dimension         = dimension,
            segment           = segment,
            origination_month = orig_month,
            performance_month = perf_month,
            at_origination    = orig_snap,
            at_performance    = perf_snap,
            macro_narrative   = narrative,
            regime_label      = regime,
        )

    def enrich_report(
        self,
        report_cards:      dict,
        dimension_registry: dict,
        top_n:             int = 3,
    ) -> dict[str, list[MacroContext]]:
        """
        Enriches the top N anomalies per dimension with macro context.

        Parameters
        ----------
        report_cards       : Output of HunterAgent.run_strategic_audit()
        dimension_registry : DIMENSION_REGISTRY from hunter.py
        top_n              : How many top anomalies per dimension to enrich

        Returns dict: dimension_key → [MacroContext, ...]
        """
        print("  🌐  Macro Agent: enriching segments with FRED macro context...")
        macro_contexts: dict[str, list[MacroContext]] = {}

        for dim_key, df in report_cards.items():
            if df is None or df.empty:
                continue

            spec       = dimension_registry.get(dim_key, {})
            dim_cols   = spec.get("dimensions", [])
            contexts   = []

            top_rows = df.head(top_n)
            for _, row in top_rows.iterrows():
                row.index = row.index.str.lower().str.strip()
                segment = {
                    col: row[col.lower()] for col in dim_cols
                    if col.lower() in row.index
                }
                if not segment:
                    continue

                ctx = self.enrich_segment(dim_key, segment, dim_cols)
                contexts.append(ctx)
                print(
                    f"     ↳ [{dim_key}] {segment} | "
                    f"orig={ctx.origination_month} | "
                    f"regime: {ctx.regime_label}"
                )

            macro_contexts[dim_key] = contexts

        return macro_contexts

    def get_macro_timeseries(
        self,
        start: str = "2012-01-01",
        end:   str = "2020-12-31",
    ) -> pd.DataFrame:
        """
        Returns the full macro panel DataFrame for dashboard charting.
        Columns = macro series names, index = YYYY-MM month strings.
        """
        df = self._load_macro_panel()
        return df.loc[start[:7]:end[:7]] if not df.empty else df