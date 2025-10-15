# ======================================================
# File: utils.py
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict, Any
import io
import clustering as clu
import evaluation as eva
# =====================================================================================
# AUTO-RECOMMENDATION
# =====================================================================================

def auto_recommend_k(X, k_range=range(2, 11)):
    silhouette_scores = []
    wcss_scores = []
    
    for k in k_range:
        labels, centroids = clu.kmeans_from_scratch(X, k, random_state=42)
        sil_score = eva.silhouette_score_from_scratch(X, labels)
        wcss = clu.calculate_wcss(X, labels, centroids)
        
        silhouette_scores.append(sil_score)
        wcss_scores.append(wcss)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return {
        'optimal_k': optimal_k,
        'k_values': list(k_range),
        'silhouette_scores': silhouette_scores,
        'wcss_scores': wcss_scores
    }

def auto_recommend_eps(X, eps_range=np.arange(0.5, 3.1, 0.3), min_samples_range=range(3, 11)):
    best_params = {'eps': None, 'min_samples': None, 'silhouette': -1}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = clu.dbscan_from_scratch(X, eps, min_samples)
            n_clusters = len([l for l in np.unique(labels) if l != -1])
            n_noise = np.sum(labels == -1)
            
            if n_clusters > 1 and n_noise < len(X) * 0.5:
                sil_score = eva.silhouette_score_from_scratch(X, labels)
                
                if sil_score > best_params['silhouette']:
                    best_params = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'silhouette': sil_score,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise
                    }
    
    return best_params

def auto_recommend_k_hierarchical(X, max_clusters=10):
    results = []
    for k in range(2, max_clusters + 1):
        labels, _ = clu.divisive_hierarchical_from_scratch(X, n_clusters=k)
        sil = eva.silhouette_score_from_scratch(X, labels)
        results.append((k, sil))
    best_k, best_score = max(results, key=lambda x: x[1])
    return {'optimal_k': best_k, 'scores': results}

# =====================================================================================
# ENHANCED INTERPRETATION
# =====================================================================================

def interpret_clusters_enhanced(X, labels, df_fe, df_original):
    """Enhanced cluster interpretation with new features."""
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    interpretations = []
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_data = X[cluster_mask]
        
        mean_score = np.mean(cluster_data)
        std_score = np.std(cluster_data)
        n_members = len(cluster_data)
        
        # Get feature engineering scores for this cluster
        s_ready_mean = df_fe.loc[cluster_mask, 's_ready'].mean() if 's_ready' in df_fe.columns else 0
        stabilitas_mean = df_fe.loc[cluster_mask, 'skor_stabilitas'].mean() if 'skor_stabilitas' in df_fe.columns else 0
        gap_mean = df_fe.loc[cluster_mask, 'gap_tinggi_rendah'].mean() if 'gap_tinggi_rendah' in df_fe.columns else 0
        
        # Enhanced interpretation logic
        if s_ready_mean > 0.5 and stabilitas_mean > 0.6:
            interpretation = "Siap Riset"
            description = "Kelompok dengan kesiapan tinggi dan konsisten, siap untuk penelitian mandiri"
            recommendation = "Arahkan untuk penelitian lanjutan dan publikasi jurnal"
        elif mean_score > 0.5 and std_score > 0.3:
            interpretation = "Antusias tapi Belum Stabil"
            description = "Memiliki potensi namun perlu konsistensi dalam pendekatan riset"
            recommendation = "Berikan mentoring intensif dan workshop praktis"
        elif mean_score <= 0.5 and stabilitas_mean > 0.5:
            interpretation = "Perlu Pendampingan"
            description = "Membutuhkan pelatihan dasar yang terstruktur dan berkelanjutan"
            recommendation = "Adakan workshop metodologi penelitian dasar"
        elif mean_score <= 0.5 and gap_mean < 0:
            interpretation = "Ragu dan Butuh Bimbingan"
            description = "Perlu perhatian khusus, mentoring intensif, dan dukungan akademik"
            recommendation = "Program pendampingan 1-on-1 dengan dosen pembimbing"
        else:
            interpretation = "Berkembang"
            description = "Dalam proses pembelajaran dan membutuhkan dukungan moderat"
            recommendation = "Fasilitasi diskusi kelompok dan peer learning"
        
        interpretations.append({
            'Cluster': f'Cluster {label}',
            'Jumlah Mahasiswa': n_members,
            'Persentase': f'{(n_members/len(labels)*100):.1f}%',
            'Mean Skor': f'{mean_score:.3f}',
            'Std Deviasi': f'{std_score:.3f}',
            'S-Ready': f'{s_ready_mean:.3f}',
            'Stabilitas': f'{stabilitas_mean:.3f}',
            'Interpretasi': interpretation,
            'Deskripsi': description,
            'Rekomendasi': recommendation
        })
    
    return pd.DataFrame(interpretations)

def get_cluster_aspect_summary(df_fe, labels):
    """Get average aspect scores per cluster."""
    aspect_cols = ['pengetahuan_riset', 'keterampilan_teknis', 'kesiapan_kepercayaan', 
                   'kebutuhan_pelatihan', 'kebiasaan_akademik', 'profil_aktif']
    
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    summary = []
    
    for label in unique_labels:
        cluster_mask = labels == label
        row = {'Cluster': f'Cluster {label}'}
        
        for col in aspect_cols:
            if col in df_fe.columns:
                row[col.replace('_', ' ').title()] = df_fe.loc[cluster_mask, col].mean()
        
        summary.append(row)
    
    return pd.DataFrame(summary)

# =====================================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================================
def plot_distribution_comparison(before_data, after_data, columns, title="Distribution Comparison"):
    """Plot before/after distributions for outlier handling with correct subplot labels."""
    n_cols = len(columns)
    subplot_titles = []
    for col in columns:
        subplot_titles.append(f"{col} - Before")
        subplot_titles.append(f"{col} - After")

    fig = make_subplots(rows=n_cols, cols=2, subplot_titles=subplot_titles)

    for i, col in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=before_data[col], name=f"{col} - Before", marker_color="#EF4444", opacity=0.7),
            row=i + 1, col=1
        )
        fig.add_trace(
            go.Histogram(x=after_data[col], name=f"{col} - After", marker_color="#10B981", opacity=0.7),
            row=i + 1, col=2
        )

    fig.update_layout(height=300 * n_cols,
        title_text=title,
        template="plotly_white",
        bargap=0.2,
        showlegend=False
    )
    return fig


def plot_scaling_comparison(before_data, after_data, feature_cols):
    """Plot distribution before and after scaling."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Before Scaling', 'After Scaling'))
    
    for col in feature_cols[:5]:  # Limit to first 5 features
        fig.add_trace(go.Box(y=before_data[col], name=col, showlegend=True), row=1, col=1)
        fig.add_trace(go.Box(y=after_data[col], name=col, showlegend=False), row=1, col=2)
    
    fig.update_layout(height=400, title_text="Feature Distribution Comparison", template='plotly_white')
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Features',
        height=600,
        template='plotly_white'
    )
    return fig

def plot_pca_variance(eigenvalues):
    """Plot PCA explained variance."""
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_variance) + 1)), y=explained_variance, 
                         name='Individual', marker_color='#1E3A8A'))
    fig.add_trace(go.Scatter(x=list(range(1, len(explained_variance) + 1)), y=cumulative_variance, 
                             name='Cumulative', line=dict(color='#D97706', width=2)))
    fig.update_layout(title='Explained Variance by Principal Components', 
                      xaxis_title='Principal Component', yaxis_title='Explained Variance Ratio', 
                      yaxis=dict(tickformat=".0%"), template='plotly_white')
    return fig

def plot_elbow_method(k_values, wcss_scores, silhouette_scores):
    """Plot Elbow Method for K-Means."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('WCSS vs K (Elbow Method)', 'Silhouette Score vs K'))
    
    fig.add_trace(go.Scatter(x=k_values, y=wcss_scores, mode='lines+markers', 
                             name='WCSS', line=dict(color='#DC2626', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_values, y=silhouette_scores, mode='lines+markers', 
                             name='Silhouette', line=dict(color='#059669', width=3)), row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
    fig.update_yaxes(title_text="WCSS", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    
    return fig

def plot_dendrogram_simple(cluster_history, n_samples):
    """Visualize hierarchical clustering progression."""
    n_steps = len(cluster_history)
    
    fig = go.Figure()
    
    for step, clusters in enumerate(cluster_history):
        y_positions = []
        x_positions = []
        sizes = []
        
        for i, cluster in enumerate(clusters):
            y_positions.append(step)
            x_positions.append(i)
            sizes.append(len(cluster))
        
        fig.add_trace(go.Scatter(
            x=x_positions, y=y_positions, mode='markers',
            marker=dict(size=[s*2 for s in sizes], color=sizes, 
                       colorscale='Viridis', showscale=(step==0)),
            text=[f'Cluster {i}<br>Size: {s}' for i, s in enumerate(sizes)],
            hoverinfo='text',
            name=f'Step {step}'
        ))
    
    fig.update_layout(
        title='Hierarchical Clustering Progression',
        xaxis_title='Cluster Index',
        yaxis_title='Division Step',
        template='plotly_white',
        showlegend=False,
        height=500
    )
    
    return fig

def plot_dbscan_results(X_pca, labels):
    """Plot DBSCAN results."""
    df_plot = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    df_plot['Cluster'] = [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='Cluster',
        title=f'DBSCAN Results (Noise Points: {np.sum(labels == -1)})',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig.update_traces(marker=dict(size=8))
    return fig

def plot_clusters_2d(X_pca, labels):
    """Plot clusters in 2D."""
    df_plot = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    df_plot['Cluster'] = [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='Cluster', 
        title='Visualisasi Cluster 2D (Hasil PCA)',
        template='plotly_white', 
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(legend_title_text='Kelompok Cluster')
    return fig