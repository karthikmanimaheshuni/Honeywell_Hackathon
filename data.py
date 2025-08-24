from __future__ import annotations
import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Generic pipeline error."""

def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "date", "datetime", "ts", "time_stamp")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(1, int(len(df) * 0.5)):
                return c
        except Exception:
            continue
    return None

def ensure_regular_intervals(ts: pd.Series) -> Tuple[bool, Optional[pd.Timedelta]]:
    diffs = ts.sort_values().diff().dropna()
    if len(diffs) == 0:
        return False, None
    mode = diffs.mode()
    if len(mode) == 0:
        return False, None
    mode_val = mode.iloc[0]
    prop = (np.abs((diffs - mode_val) / (mode_val + pd.Timedelta(seconds=1))) < 0.01).mean()
    return prop > 0.8, pd.Timedelta(mode_val)

def percentile_score(arr: np.ndarray, scale_min: float = 0.0, scale_max: float = 100.0) -> np.ndarray:
    series = pd.Series(arr)
    ranks = series.rank(method="average", pct=True).values
    return (ranks * (scale_max - scale_min) + scale_min).astype(float)

@dataclass
class DataProcessor:
    timestamp_col: Optional[str] = None

    def load_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise PipelineError(f"Input file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise PipelineError(f"Failed to read CSV: {e}")
        if df.shape[0] == 0:
            raise PipelineError("Input CSV is empty.")
        return df

    def detect_and_parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_col = self.timestamp_col or find_timestamp_column(df)
        if ts_col is None:
            raise PipelineError("Could not identify a timestamp column.")
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        except Exception as e:
            raise PipelineError(f"Failed parsing timestamps: {e}")
        if df[ts_col].isna().any():
            frac_bad = df[ts_col].isna().mean()
            if frac_bad > 0.2:
                raise PipelineError("Timestamp column contains many unparsable values.")
            df = df.dropna(subset=[ts_col]).copy()
        df = df.sort_values(ts_col).reset_index(drop=True)
        self.timestamp_col = ts_col
        regular, interval = ensure_regular_intervals(df[ts_col])
        if not regular:
            logger.warning("Timestamps may not be regular intervals.")
        return df

    def coerce_and_fill(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        ts = self.timestamp_col
        feature_cols = [c for c in df.columns if c != ts]
        if not feature_cols:
            raise PipelineError("No feature columns found (only timestamp present).")
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Data quality: forward-fill, interpolate, then backfill
        df[feature_cols] = df[feature_cols].ffill().interpolate(limit_direction="both").bfill()
        feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        if not feature_cols:
            raise PipelineError("All feature columns non-numeric or empty after coercion.")
        # Remove constant (zero-variance) features
        nunique = df[feature_cols].nunique()
        constant_features = nunique[nunique <= 1].index.tolist()
        model_features = [c for c in feature_cols if c not in constant_features]
        if not model_features:
            raise PipelineError("No variable features for modeling after removing zero-variance.")
        return df, model_features

class PCAAnomalyDetector:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_names: List[str] = []
        self.trained = False
        self.means = None
        self.stds = None

    def fit(self, X: pd.DataFrame) -> None:
        if X.shape[0] < 2:
            raise ValueError("Training data must have at least 2 rows.")
        self.feature_names = X.columns.tolist()
        Xs = X.values.astype(float)
        self.scaler = StandardScaler()
        Xs_scaled = self.scaler.fit_transform(Xs)
        # Keep means & stds for explicit threshold logic
        self.means = np.mean(Xs_scaled, axis=0)
        self.stds = np.std(Xs_scaled, axis=0)
        if self.n_components is None:
            pca_full = PCA(n_components=min(len(self.feature_names), Xs_scaled.shape[0] - 1))
            pca_full.fit(Xs_scaled)
            cum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(cum, 0.95) + 1)
            k = max(1, min(k, pca_full.n_components_))
            self.pca = PCA(n_components=k)
            self.pca.fit(Xs_scaled)
        else:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(Xs_scaled)
        self.trained = True
        logger.info("PCA trained: components=%d", self.pca.n_components_)

    def reconstruction_error(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained or self.pca is None or self.scaler is None:
            raise RuntimeError("Detector not trained yet.")
        Xs = X.values.astype(float)
        Xs_scaled = self.scaler.transform(Xs)
        transformed = self.pca.transform(Xs_scaled)
        reconstructed = self.pca.inverse_transform(transformed)
        per_feature_error = np.abs(Xs_scaled - reconstructed)
        sample_error = per_feature_error.sum(axis=1)
        return per_feature_error, sample_error

    def threshold_violations(self, X: pd.DataFrame) -> List[List[str]]:
        violations_idx = []
        Xs_scaled = self.scaler.transform(X.values.astype(float))
        for i in range(Xs_scaled.shape[0]):
            violated = []
            for j, (m, s, name) in enumerate(zip(self.means, self.stds, self.feature_names)):
                if abs(Xs_scaled[i, j] - m) > 3 * s:
                    violated.append(name)
            violations_idx.append(violated)
        return violations_idx

    def correlation_changes(self, X: pd.DataFrame, train_corr: pd.DataFrame, tolerance=0.5) -> List[List[str]]:
        corrs = X.corr()
        changed_pairs = []
        for c1 in train_corr.columns:
            for c2 in train_corr.columns:
                if c1 >= c2:
                    continue
                diff = abs(corrs.loc[c1, c2] - train_corr.loc[c1, c2])
                if diff > tolerance:
                    changed_pairs.append((c1, c2))
        # Report features suspicious from broken correlation
        flagged = []
        for i in range(X.shape[0]):
            flags = list(set([f for pair in changed_pairs for f in pair]))
            flagged.append(flags)
        return flagged

def top_contributors(
    per_feature_error_row: np.ndarray,
    feature_names: List[str],
    threshold_pct: float = 0.01,
    top_k: int = 7,
) -> List[str]:
    total = float(per_feature_error_row.sum())
    if total <= 0:
        return []
    contribs = per_feature_error_row / total
    items = list(zip(feature_names, contribs))
    # Only include if > threshold
    items = [it for it in items if it[1] > threshold_pct]
    items_sorted = sorted(items, key=lambda x: (-x[1], x))
    top = [name for name, _ in items_sorted[:top_k]]
    return top

def label_anomaly_type(row_idx, threshold_flags, corr_flags, score, degree_breaks):
    """Choose anomaly type according to detection logic."""
    if threshold_flags[row_idx]:
        return "ThresholdViolation"
    elif corr_flags[row_idx]:
        return "RelationshipChange"
    elif score >= degree_breaks[2]:
        return "PatternDeviation"
    else:
        return "Normal"

def process(
    input_csv: str,
    output_csv: str,
    timestamp_col: Optional[str] = None,
    train_start: str = "2004-01-01 00:00",
    train_end: str = "2004-01-05 23:59",
    analysis_start: str = "2004-01-01 00:00",
    analysis_end: str = "2004-01-19 07:59",
) -> None:
    dp = DataProcessor(timestamp_col=timestamp_col)
    df = dp.load_csv(input_csv)
    df = dp.detect_and_parse_timestamp(df)
    df, model_features = dp.coerce_and_fill(df)
    ts_col = dp.timestamp_col

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    analysis_start_dt = pd.to_datetime(analysis_start)
    analysis_end_dt = pd.to_datetime(analysis_end)
    mask_analysis = (df[ts_col] >= analysis_start_dt) & (df[ts_col] <= analysis_end_dt)
    df_analysis = df.loc[mask_analysis].reset_index(drop=True)
    if df_analysis.shape[0] == 0:
        raise PipelineError("No data found in the analysis period.")
    mask_train = (df_analysis[ts_col] >= train_start_dt) & (df_analysis[ts_col] <= train_end_dt)
    df_train = df_analysis.loc[mask_train, model_features]
    if df_train.shape[0] < max(10, len(model_features)):
        logger.warning("Training rows (%d) small for feature number (%d).", df_train.shape, len(model_features))

    detector = PCAAnomalyDetector()
    detector.fit(df_train)

    per_feat_err, sample_err = detector.reconstruction_error(df_analysis[model_features])
    scores = percentile_score(sample_err, 0.0, 100.0)
    train_indices = df_analysis.index[mask_train.values]
    if len(train_indices) > 0:
        train_mean = float(scores[train_indices].mean())
        train_max = float(scores[train_indices].max())
        if train_mean >= 10 or train_max >= 25:
            logger.info("Training scores high, scaling for conservative output.")
            p75 = np.percentile(sample_err[train_indices], 75)
            scaling = 10.0 / max(1e-6, p75 if p75 > 0 else 0.1)
            scores = np.clip(scores * scaling, 0.0, 100.0)

    # Degree breaks for explanation
    degree_breaks = [10, 30, 60, 90]

    # Threshold Violations
    threshold_flags = detector.threshold_violations(df_analysis[model_features])
    # Correlation Changes (relationship change, global for analysis block)
    train_corr = df_train.corr()
    corr_flags = detector.correlation_changes(df_analysis[model_features], train_corr, tolerance=0.5)

    top_features_rows: List[List[str]] = []
    anomaly_types: List[str] = []
    for i in range(per_feat_err.shape[0]):
        row_err = per_feat_err[i]
        tops = top_contributors(row_err, model_features, threshold_pct=0.01, top_k=7)
        while len(tops) < 7:
            tops.append("")
        top_features_rows.append(tops)
        anomaly_type = label_anomaly_type(i, threshold_flags, corr_flags, scores[i], degree_breaks)
        anomaly_types.append(anomaly_type)

    out_df = df_analysis.copy().reset_index(drop=True)
    out_df["Abnormality_score"] = np.round(scores, 3)
    for idx in range(7):
        out_df[f"top_feature_{idx+1}"] = [r[idx] for r in top_features_rows]
    out_df["Anomaly_Type"] = anomaly_types

    new_cols = ["Abnormality_score"] + [f"top_feature_{i+1}" for i in range(7)] + ["Anomaly_Type"]
    if not all(c in out_df.columns for c in new_cols):
        raise PipelineError("Output columns missing.")
    try:
        out_df.to_csv(output_csv, index=False)
        logger.info("Saved processed CSV to %s", output_csv)
    except Exception as e:
        raise PipelineError(f"Failed to save CSV: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multivariate Time Series Anomaly Detection Full Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--timestamp", "-t", default=None, help="Timestamp column name (auto-detected if omitted)")
    parser.add_argument("--train-start", default="2004-01-01 00:00", help="Training period start")
    parser.add_argument("--train-end", default="2004-01-05 23:59", help="Training period end")
    parser.add_argument("--analysis-start", default="2004-01-01 00:00", help="Analysis window start")
    parser.add_argument("--analysis-end", default="2004-01-19 07:59", help="Analysis window end")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        process(
            input_csv=args.input,
            output_csv=args.output,
            timestamp_col=args.timestamp,
            train_start=args.train_start,
            train_end=args.train_end,
            analysis_start=args.analysis_start,
            analysis_end=args.analysis_end,
        )
    except PipelineError as e:
        logger.error("PipelineError: %s", e)
        raise SystemExit(2)
    except Exception as e:
        logger.exception("Unhandled exception during processing: %s", e)
        raise SystemExit(3)

if __name__ == "__main__":
    main()
