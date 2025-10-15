# ======================================================
# File: evaluation.py
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
# =====================================================================================
# EVALUATION METRICS
# =====================================================================================

def silhouette_score_from_scratch(X, labels):
    n_samples = len(X)
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return 0

    silhouette_vals = []
    for i in range(n_samples):
        own_cluster = labels[i]
        if own_cluster == -1:
            continue
        
        own_cluster_points = X[labels == own_cluster]
        if len(own_cluster_points) <= 1:
            continue
        a_i = np.mean([clu.euclidean_distance(X[i], p) for p in own_cluster_points if not np.array_equal(p, X[i])])

        b_i = np.inf
        for label in unique_labels:
            if label == own_cluster:
                continue
            other_cluster_points = X[labels == label]
            mean_dist = np.mean([clu.euclidean_distance(X[i], p) for p in other_cluster_points])
            if mean_dist < b_i:
                b_i = mean_dist
        
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
            silhouette_vals.append(s_i)

    return np.mean(silhouette_vals) if silhouette_vals else 0

def calinski_harabasz_score_from_scratch(X, labels):
    n_samples = X.shape[0]
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return 0

    overall_mean = np.mean(X, axis=0)
    ssb = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_c = len(cluster_points)
        cluster_mean = np.mean(cluster_points, axis=0)
        ssb += n_c * np.sum((cluster_mean - overall_mean) ** 2)

    ssw = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        ssw += np.sum((cluster_points - cluster_mean) ** 2)

    if ssw == 0:
        return 1.0 
    
    ch_score = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))
    return ch_score

def davies_bouldin_score_from_scratch(X, labels):
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0

    centroids = [np.mean(X[labels == l], axis=0) for l in unique_labels]
    s = np.array([np.mean([clu.euclidean_distance(p, centroids[i]) for p in X[labels == l]]) for i, l in enumerate(unique_labels)])

    db_score = 0
    for i in range(n_clusters):
        max_rij = 0
        for j in range(n_clusters):
            if i == j:
                continue
            m_ij = clu.euclidean_distance(centroids[i], centroids[j])
            if m_ij > 0:
                r_ij = (s[i] + s[j]) / m_ij
                if r_ij > max_rij:
                    max_rij = r_ij
        db_score += max_rij

    return db_score / n_clusters

def dunn_index_from_scratch(X, labels):
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0
    
    min_inter_cluster = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            min_dist = np.min([clu.euclidean_distance(pi, pj) for pi in cluster_i for pj in cluster_j])
            if min_dist < min_inter_cluster:
                min_inter_cluster = min_dist
    
    max_intra_cluster = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            max_dist = np.max([clu.euclidean_distance(pi, pj) for i, pi in enumerate(cluster_points) for pj in cluster_points[i+1:]])
            if max_dist > max_intra_cluster:
                max_intra_cluster = max_dist
    
    if max_intra_cluster == 0:
        return 0
    
    return min_inter_cluster / max_intra_cluster

def compactness_ratio_from_scratch(X, labels):
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    if len(unique_labels) <= 1:
        return 1.0
    
    overall_mean = np.mean(X, axis=0)
    total_variance = np.sum((X - overall_mean) ** 2)
    
    within_variance = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        within_variance += np.sum((cluster_points - cluster_mean) ** 2)
    
    if total_variance == 0:
        return 0
    
    return within_variance / total_variance

def cluster_balance_ratio(labels):
    """Calculate balance ratio of cluster sizes."""
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) <= 1:
        return 1.0
    
    sizes = [np.sum(labels == l) for l in unique_labels]
    return np.min(sizes) / np.max(sizes) if np.max(sizes) > 0 else 0

def entropy_index(labels, categorical_feature):
    """Calculate entropy of cluster distribution by categorical feature."""
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) <= 1:
        return 0
    
    total_entropy = 0
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_cats = categorical_feature[cluster_mask]
        
        unique_cats, counts = np.unique(cluster_cats, return_counts=True)
        probs = counts / len(cluster_cats)
        cluster_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        total_entropy += (len(cluster_cats) / len(labels)) * cluster_entropy
    
    return total_entropy