"""
detectors.py — Algorithm registry for anomaly detection.

Four algorithms, one interface. Each detector takes a DataFrame
of grouped metric values and returns anomaly scores + flagged rows.

Usage:
    from agents.detectors import get_detector
    detector = get_detector("zscore", threshold=2.0)
    anomalies = detector.detect(df, metric_col="metric_value")

Algorithms:
    zscore           — flags segments > N standard deviations from mean
    iqr              — flags segments outside IQR fence (robust to outliers)
    isolation_forest — tree-based, good for non-linear anomalies
    lof              — density-based, good for local cluster anomalies

Note: All algorithms operate on cross-sectional grouped data.
      Time series anomaly detection (Prophet, Matrix Profile) is
      on the roadmap but not supported in this version.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Abstract base for all anomaly detectors.

    Every detector receives a DataFrame where each row is a grouped segment
    (e.g. addr_state=DC with avg_dti=15.7) and returns the same DataFrame
    with two additional columns:
        anomaly_score : float — higher = more anomalous
        z_score       : float — standardised score for cross-algorithm comparison
        is_anomaly    : bool  — True if flagged as anomalous
    """

    def __init__(self, threshold: float = 2.0, min_sample_size: int = 100):
        self.threshold       = threshold
        self.min_sample_size = min_sample_size

    @abstractmethod
    def detect(
        self,
        df:         pd.DataFrame,
        metric_col: str = "metric_value",
        top_n:      int = 10,
    ) -> pd.DataFrame:
        """
        Detect anomalies in grouped metric values.

        Parameters
        ----------
        df         : DataFrame with grouped segments and metric values.
                     Must contain metric_col and optionally 'volume'.
        metric_col : Column name containing the metric to analyse.
        top_n      : Maximum anomalies to return.

        Returns
        -------
        DataFrame of anomalous rows sorted by severity, with columns:
            anomaly_score, z_score, is_anomaly
        """
        pass

    def _add_zscore_column(self, df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        """Adds a standardised z_score column for cross-algorithm comparison."""
        mean = df[metric_col].mean()
        std  = df[metric_col].std()
        df   = df.copy()
        if std > 0:
            df["z_score"] = ((df[metric_col] - mean) / std).round(2)
        else:
            df["z_score"] = 0.0
        df["avg_val"] = round(mean, 4)
        df["std_val"] = round(std,  4)
        return df

    def _filter_min_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes groups below min_sample_size if volume column exists."""
        if "volume" in df.columns:
            return df[df["volume"] >= self.min_sample_size].copy()
        return df.copy()

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Detector", "").lower()


# ---------------------------------------------------------------------------
# Z-Score detector
# ---------------------------------------------------------------------------

class ZScoreDetector(BaseDetector):
    """
    Flags segments whose metric value is more than `threshold` standard
    deviations from the population mean.

    Best for: normally distributed metrics, easy to interpret, fast.
    Limitation: sensitive to outliers in the population mean/std.
    """

    def detect(
        self,
        df:         pd.DataFrame,
        metric_col: str = "metric_value",
        top_n:      int = 10,
    ) -> pd.DataFrame:
        df = self._filter_min_sample(df)
        if df.empty:
            return df

        df = self._add_zscore_column(df, metric_col)
        df["anomaly_score"] = df["z_score"].abs()
        df["is_anomaly"]    = df["anomaly_score"] > self.threshold

        return (
            df[df["is_anomaly"]]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# IQR detector
# ---------------------------------------------------------------------------

class IQRDetector(BaseDetector):
    """
    Flags segments outside the IQR fence:
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR

    threshold here is the IQR multiplier (default 1.5 = standard Tukey fence,
    3.0 = far outliers only).

    Best for: skewed distributions, robust to extreme values.
    Limitation: less sensitive than z-score for normal distributions.
    """

    def detect(
        self,
        df:         pd.DataFrame,
        metric_col: str = "metric_value",
        top_n:      int = 10,
    ) -> pd.DataFrame:
        df = self._filter_min_sample(df)
        if df.empty:
            return df

        q1  = df[metric_col].quantile(0.25)
        q3  = df[metric_col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr

        df = self._add_zscore_column(df, metric_col)

        # IQR distance as anomaly score (normalised)
        def iqr_score(x):
            if x < lower:
                return (lower - x) / (iqr if iqr > 0 else 1)
            elif x > upper:
                return (x - upper) / (iqr if iqr > 0 else 1)
            return 0.0

        df          = df.copy()
        df["anomaly_score"] = df[metric_col].apply(iqr_score)
        df["is_anomaly"]    = (df[metric_col] < lower) | (df[metric_col] > upper)
        df["iqr_lower"]     = round(lower, 4)
        df["iqr_upper"]     = round(upper, 4)

        return (
            df[df["is_anomaly"]]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Isolation Forest detector
# ---------------------------------------------------------------------------

class IsolationForestDetector(BaseDetector):
    """
    Tree-based algorithm that isolates anomalies by randomly partitioning
    the feature space. Anomalies are isolated in fewer splits.

    threshold here is contamination — expected proportion of anomalies
    (0.05 = expect 5% of segments to be anomalous).

    Best for: non-linear relationships, multivariate anomalies,
              doesn't assume normal distribution.
    Limitation: less interpretable, requires scikit-learn.
    """

    def detect(
        self,
        df:         pd.DataFrame,
        metric_col: str = "metric_value",
        top_n:      int = 10,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "scikit-learn required for Isolation Forest. "
                "Run: pip install scikit-learn"
            )

        df = self._filter_min_sample(df)
        if df.empty or len(df) < 5:
            return df

        # Use metric_value + volume as features if available
        feature_cols = [metric_col]
        if "volume" in df.columns:
            feature_cols.append("volume")

        X = df[feature_cols].fillna(df[feature_cols].median())

        # threshold is contamination for isolation forest
        contamination = min(max(self.threshold, 0.01), 0.5)

        clf = IsolationForest(
            contamination = contamination,
            random_state  = 42,
            n_estimators  = 100,
        )
        clf.fit(X)

        df = df.copy()
        df["anomaly_score"] = -clf.score_samples(X)   # higher = more anomalous
        df["is_anomaly"]    = clf.predict(X) == -1     # -1 = anomaly

        # Add standardised z_score for cross-algorithm compatibility
        df = self._add_zscore_column(df, metric_col)

        return (
            df[df["is_anomaly"]]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Local Outlier Factor detector
# ---------------------------------------------------------------------------

class LOFDetector(BaseDetector):
    """
    Density-based algorithm that compares local density of a segment
    to its neighbours. Points in low-density regions relative to
    neighbours are flagged as anomalous.

    threshold here is contamination (same as Isolation Forest).

    Best for: detecting local anomalies in clusters,
              works well when anomalies have different densities.
    Limitation: computationally expensive on large datasets,
                requires scikit-learn, sensitive to n_neighbors.
    """

    def __init__(
        self,
        threshold:       float = 0.05,
        min_sample_size: int   = 100,
        n_neighbors:     int   = 5,
    ):
        super().__init__(threshold, min_sample_size)
        self.n_neighbors = n_neighbors

    def detect(
        self,
        df:         pd.DataFrame,
        metric_col: str = "metric_value",
        top_n:      int = 10,
    ) -> pd.DataFrame:
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError(
                "scikit-learn required for LOF. "
                "Run: pip install scikit-learn"
            )

        df = self._filter_min_sample(df)
        if df.empty or len(df) < self.n_neighbors + 1:
            print(
                f"  ⚠️  LOF requires at least {self.n_neighbors + 1} groups "
                f"(got {len(df)}) — falling back to z-score."
            )
            return ZScoreDetector(
                threshold=2.0, min_sample_size=self.min_sample_size
            ).detect(df, metric_col, top_n)

        feature_cols = [metric_col]
        if "volume" in df.columns:
            feature_cols.append("volume")

        X = df[feature_cols].fillna(df[feature_cols].median())

        contamination = min(max(self.threshold, 0.01), 0.5)
        n_neighbors   = min(self.n_neighbors, len(df) - 1)

        clf = LocalOutlierFactor(
            n_neighbors   = n_neighbors,
            contamination = contamination,
        )
        predictions = clf.fit_predict(X)

        df = df.copy()
        df["anomaly_score"] = -clf.negative_outlier_factor_
        df["is_anomaly"]    = predictions == -1

        df = self._add_zscore_column(df, metric_col)

        return (
            df[df["is_anomaly"]]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Registry — single entry point
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY = {
    "zscore":           ZScoreDetector,
    "iqr":              IQRDetector,
    "isolation_forest": IsolationForestDetector,
    "lof":              LOFDetector,
}


def get_detector(
    algorithm:       str   = "zscore",
    threshold:       float = 2.0,
    min_sample_size: int   = 100,
    **kwargs,
) -> BaseDetector:
    """
    Factory function — returns the right detector for the given algorithm name.

    Parameters
    ----------
    algorithm       : One of zscore | iqr | isolation_forest | lof
    threshold       : Z-score cutoff (zscore/iqr) or contamination % (if/lof)
    min_sample_size : Minimum group size to include in scan
    **kwargs        : Additional algorithm-specific parameters

    Example
    -------
    detector = get_detector("isolation_forest", threshold=0.05)
    anomalies = detector.detect(grouped_df, metric_col="avg_dti")
    """
    algo = algorithm.lower().strip()
    if algo not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {list(DETECTOR_REGISTRY.keys())}"
        )
    cls = DETECTOR_REGISTRY[algo]

    # LOF accepts n_neighbors kwarg
    if algo == "lof" and "n_neighbors" in kwargs:
        return cls(threshold=threshold, min_sample_size=min_sample_size,
                   n_neighbors=kwargs["n_neighbors"])

    return cls(threshold=threshold, min_sample_size=min_sample_size)
