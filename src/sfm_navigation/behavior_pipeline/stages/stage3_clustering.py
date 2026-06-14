"""Stage 3: Domain-specific clustering and regime labeling."""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from ...behavior_pipeline.config import PipelineConfig
from ...behavior_pipeline.reporting import PipelineLogger

# ------------------------------------------------------------
# Domain definitions
# ------------------------------------------------------------
DOMAINS = {
    'kinematic': {
        'cols': ['speed_mean', 'speed_std', 'stop_ratio', 'accel_mean', 'accel_max',
                 'accel_std', 'jerk_mean', 'jerk_max'],
        'label_col': 'kinematic_cluster',
        'title': 'Kinematic Clusters'
    },
    'path': {
        'cols': ['tortuosity', 'mean_curvature', 'max_curvature', 'sinuosity',
                 'spectral_arc_length', 'mean_step_angle', 'step_angle_std'],
        'label_col': 'path_cluster',
        'title': 'Path Quality Clusters'
    },
    'safety': {
        'cols': ['min_distance', 'min_TTC', 'intrusion_time_frac', 'num_intrusions',
                 'avoidance_deviation', 'n_neighbors'],
        'label_col': 'safety_cluster',
        'title': 'Safety / Interaction Clusters'
    },
    'social': {
        'cols': ['fov_occupancy', 'mutual_attention_count', 'o_space_ratio', 'group_affiliation'],
        'label_col': 'social_cluster',
        'title': 'Social Attention Clusters'
    }
}

LOG_COLS = [
    'mean_curvature', 'max_curvature', 'jerk_mean', 'jerk_max',
    'tortuosity', 'avoidance_deviation', 'intrusion_time_frac',
    'num_intrusions', 'min_TTC', 'min_distance',
    'stop_ratio', 'accel_mean', 'accel_max', 'accel_std',
    'sinuosity', 'mean_step_angle', 'step_angle_std'
]

# ------------------------------------------------------------
# Helper: merge features
# ------------------------------------------------------------
def _merge_features(f11, f12, f13, f14):
    df = f11.merge(f12[['window_id', 'tortuosity', 'mean_curvature', 'max_curvature',
                         'sinuosity', 'spectral_arc_length', 'mean_step_angle', 'step_angle_std']],
                   on='window_id', how='inner')
    df = df.merge(f13[['window_id', 'min_distance', 'min_TTC', 'intrusion_time_frac',
                       'num_intrusions', 'avoidance_deviation', 'n_neighbors']],
                  on='window_id', how='inner')
    df = df.merge(f14[['window_id', 'fov_occupancy', 'mutual_attention_count',
                       'o_space_ratio', 'group_affiliation']],
                  on='window_id', how='inner')
    return df

# ------------------------------------------------------------
# Main stage function
# ------------------------------------------------------------
def run_stage3(config: PipelineConfig, logger: PipelineLogger,
               f11: pd.DataFrame, f12: pd.DataFrame,
               f13: pd.DataFrame, f14: pd.DataFrame):
    logger.info("=== Stage 3: Clustering & Regime Labeling ===")
    data_dir = os.path.join(logger.output_dir, logger.run_id, "data/stage3")
    plot_dir = os.path.join(logger.output_dir, logger.run_id, "reports/stage3/plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Merge features
    df = _merge_features(f11, f12, f13, f14)
    logger.info(f"Merged dataset: {df.shape}")

    # Log-transform skewed features
    for col in LOG_COLS:
        if col in df.columns:
            if df[col].min() >= 0:
                df[col] = np.log1p(df[col])
            else:
                df[col] = np.log1p(df[col] - df[col].min())

    # Store labels
    labels_df = pd.DataFrame({'window_id': df['window_id'], 'agent_id': df['agent_id']})

    report_lines = []
    for domain, info in DOMAINS.items():
        cols = info['cols']
        label_col = info['label_col']
        logger.info(f"  Clustering {domain} ({len(cols)} features)")

        # Impute, scale, PCA
        X = df[cols].copy()
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
        pca = PCA(n_components=0.90)
        X_pca = pca.fit_transform(X_scaled)
        logger.info(f"    PCA: {X_pca.shape[1]} components, var {pca.explained_variance_ratio_.sum():.2f}")
        # Save PCA loadings for feature importance analysis
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=cols
        )
        loadings.to_csv(os.path.join(data_dir, f"{domain}_pca_loadings.csv"))
        logger.info(f"    PCA loadings saved to {domain}_pca_loadings.csv")
        
        # Cluster with KMeans or GMM
        best_k, best_sil = 2, -1
        if config.cluster_method == "gmm":
            for k in range(2, min(6, len(df)//20 + 1)):
                gmm = GaussianMixture(n_components=k, covariance_type=config.gmm_covariance_type,
                                      random_state=42, n_init=3)
                labels = gmm.fit_predict(X_pca)
                sil = silhouette_score(X_pca, labels)
                logger.info(f"      GMM k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil, best_k = sil, k
                    best_model = gmm
        else:  # kmeans
            for k in range(2, min(6, len(df)//20 + 1)):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_pca)
                sil = silhouette_score(X_pca, labels)
                logger.info(f"      KMeans k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil, best_k = sil, k
                    best_model = km

        logger.info(f"    Best k={best_k} (silhouette {best_sil:.3f})")
        cluster_labels = best_model.predict(X_pca)
        labels_df[label_col] = cluster_labels

        # PCA 2D plot
        pca2d = PCA(n_components=2)
        X_2d = pca2d.fit_transform(X_scaled)
        plot_df = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
        plot_df['cluster'] = cluster_labels.astype(str)
        fig = px.scatter(plot_df, x='PC1', y='PC2', color='cluster', title=info['title'], opacity=0.7)
        fig.write_html(os.path.join(plot_dir, f"{domain}_clusters.html"))

        # Cluster means (original scale)
        orig_cols = [c for c in cols if c in df.columns]
        means = df[orig_cols].groupby(cluster_labels).mean()
        report_lines.append(f"\n{domain} cluster means (original scale):\n{means.round(3).to_string()}")

    # Save domain cluster labels
    labels_df.to_csv(os.path.join(data_dir, "domain_clusters.csv"), index=False)
    logger.info("Domain cluster labels saved.")

    # ---------------------------------------------------------
    # Phase 2d: Combine domain labels into regimes
    # ---------------------------------------------------------
    logger.info("  Combining domain labels into walking regimes...")
    label_cols = [d['label_col'] for d in DOMAINS.values()]
    labels_df['regime_code'] = ''
    for col in label_cols:
        labels_df['regime_code'] += labels_df[col].astype(str) + '_'
    labels_df['regime_code'] = labels_df['regime_code'].str.rstrip('_')

    regime_counts = labels_df['regime_code'].value_counts()
    rare_regimes = regime_counts[regime_counts < config.min_windows_for_regime].index
    labels_df['regime'] = labels_df['regime_code']
    labels_df.loc[labels_df['regime_code'].isin(rare_regimes), 'regime'] = 'rare_mixed'
    unique_regimes = labels_df['regime'].unique()
    regime_to_id = {r: i for i, r in enumerate(unique_regimes)}
    labels_df['regime_id'] = labels_df['regime'].map(regime_to_id)

    logger.info(f"Final regimes: {len(unique_regimes)} (including 'rare_mixed')")
    labels_df[['window_id', 'agent_id', 'regime', 'regime_id', 'regime_code']].to_csv(
        os.path.join(data_dir, "regime_labels.csv"), index=False)

    # Merge back original (non-log) feature values for regime characterization
    df_orig = _merge_features(f11, f12, f13, f14)
    df_orig = df_orig.merge(labels_df[['window_id', 'regime']], on='window_id', how='inner')
    key_features = [
        'speed_mean', 'speed_std', 'stop_ratio',
        'tortuosity', 'mean_curvature', 'max_curvature', 'sinuosity',
        'intrusion_time_frac', 'num_intrusions', 'avoidance_deviation',
        'fov_occupancy', 'mutual_attention_count', 'group_affiliation'
    ]
    regime_means = df_orig.groupby('regime')[key_features].mean()
    regime_means.to_csv(os.path.join(data_dir, "regime_means.csv"))

    # Bar chart of regime distribution
    regime_counts_plot = labels_df['regime'].value_counts().reset_index()
    regime_counts_plot.columns = ['Regime', 'Count']
    fig = px.bar(regime_counts_plot, x='Regime', y='Count', title='Walking Regime Distribution')
    fig.write_html(os.path.join(plot_dir, "regime_distribution.html"))

    report = "\n".join(report_lines)
    report += f"\n\nRegime distribution:\n{labels_df['regime'].value_counts().to_string()}"
    logger.write_stage_report("stage3", report)
    logger.info("Stage 3 complete.")
    return labels_df, regime_means