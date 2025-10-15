# ======================================================
# File: clustering.py
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict, Any
import io
# =====================================================================================
# CLUSTERING ALGORITHMS
# =====================================================================================

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def kmeans_from_scratch(X, k, max_iters=100, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]
    
    for _ in range(max_iters):
        labels = np.array([np.argmin([euclidean_distance(point, c) for c in centroids]) for point in X])
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return labels, centroids

def calculate_wcss(X, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def dbscan_from_scratch(X, eps, min_samples):
    n_points = X.shape[0]
    labels = np.full(n_points, -1)
    cluster_id = 0
    visited = np.zeros(n_points, dtype=bool)
    
    dist_matrix = np.array([[euclidean_distance(X[i], X[j]) for j in range(n_points)] for i in range(n_points)])
    
    for i in range(n_points):
        if visited[i]:
            continue
            
        visited[i] = True
        neighbors = np.where(dist_matrix[i] < eps)[0]
        
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        
        labels[i] = cluster_id
        seed_set = list(neighbors)
        
        idx = 0
        while idx < len(seed_set):
            point_idx = seed_set[idx]
            
            if not visited[point_idx]:
                visited[point_idx] = True
                point_neighbors = np.where(dist_matrix[point_idx] < eps)[0]
                
                if len(point_neighbors) >= min_samples:
                    seed_set.extend([n for n in point_neighbors if n not in seed_set])
            
            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id
                
            idx += 1
        
        cluster_id += 1
    
    return labels

def divisive_hierarchical_from_scratch(X, n_clusters):
    def get_sse(points):
        if len(points) == 0:
            return 0
        centroid = np.mean(points, axis=0)
        return np.sum([euclidean_distance(p, centroid)**2 for p in points])

    clusters = [list(range(X.shape[0]))]
    cluster_history = [clusters.copy()]

    while len(clusters) < n_clusters:
        max_sse = -1
        split_candidate_idx = -1
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 1:
                sse = get_sse(X[cluster_indices])
                if sse > max_sse:
                    max_sse = sse
                    split_candidate_idx = i

        if split_candidate_idx == -1:
            break

        indices_to_split = clusters[split_candidate_idx]
        data_to_split = X[indices_to_split]

        labels_split, _ = kmeans_from_scratch(data_to_split, k=2, random_state=42)

        new_cluster_1_indices = [indices_to_split[i] for i, label in enumerate(labels_split) if label == 0]
        new_cluster_2_indices = [indices_to_split[i] for i, label in enumerate(labels_split) if label == 1]

        del clusters[split_candidate_idx]
        clusters.append(new_cluster_1_indices)
        clusters.append(new_cluster_2_indices)
        cluster_history.append(clusters.copy())

    final_labels = np.zeros(X.shape[0], dtype=int)
    for i, cluster_indices in enumerate(clusters):
        final_labels[cluster_indices] = i
        
    return final_labels, cluster_history