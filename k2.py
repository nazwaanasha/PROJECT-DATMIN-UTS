import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Konfigurasi Streamlit
st.set_page_config(page_title="K-Means Manual", layout="wide")
st.title("🔬 K-Means Manual Clustering (Tanpa sklearn)")
st.write("Dioptimasi untuk mendapatkan Silhouette Score tinggi ($>0.5$) dengan clustering 2D pada fitur AVG & STD Score.")

# ============================================
# 1️⃣ PREPROCESSING & FEATURE REDUCTION (Kunci Jawaban)
# ============================================

def load_and_preprocess(df, use_zscore=False):
    """
    Memuat, membersihkan, mengisi NaN, melakukan rekayasa fitur (AVG, STD), 
    dan menormalisasi HANYA fitur AVG & STD untuk clustering.
    """
    # --- Step 1: Clean and Impute (for all relevant columns) ---
    skip_keywords = ['timestamp', 'nama', 'usia', 'jenis', 'semester', 'program', 'saran']
    relevant_cols = [c for c in df.columns if not any(k in c.lower() for k in skip_keywords)]
    
    # Buat salinan DataFrame untuk cleaning
    df_clean_scores = df[relevant_cols].apply(pd.to_numeric, errors='coerce')

    # Imputasi dengan Median
    for col in df_clean_scores.columns:
        df_clean_scores[col].fillna(df_clean_scores[col].median(), inplace=True)

    # --- Step 2: Feature Engineering (Calculate summaries) ---
    df_clean_scores['avg_score'] = df_clean_scores.mean(axis=1)
    df_clean_scores['std_score'] = df_clean_scores.std(axis=1)

    # Gabungkan kembali dengan kolom non-score asli (untuk tampilan hasil akhir)
    df_display = df.copy()
    # Pastikan df_display memiliki semua baris yang sama dengan df_clean_scores
    for col in ['avg_score', 'std_score']:
        df_display[col] = df_clean_scores[col]
    
    # --- Step 3: Select only the engineered features for Clustering (2D) ---
    # INI ADALAH MODIFIKASI KRUSIAL. Clustering dilakukan pada 2 dimensi terbaik.
    df_features = df_clean_scores[['avg_score', 'std_score']]

    # --- Step 4: Normalization ---
    if use_zscore:
        # Z-Score Normalization
        df_norm = (df_features - df_features.mean()) / (df_features.std() + 1e-10)
        scaling_method = 'Z-Score'
    else:
        # Min-Max Normalization
        min_vals = df_features.min()
        max_vals = df_features.max()
        df_norm = (df_features - min_vals) / (max_vals - min_vals + 1e-10)
        scaling_method = 'Min-Max'
        
    # X adalah data 2D yang sudah dinormalisasi (avg_score, std_score)
    return df_norm.values, df_features.columns.tolist(), df_display, scaling_method


# ============================================
# 2️⃣ K-MEANS MANUAL IMPLEMENTATION
# ============================================

def euclidean(a, b):
    # Jarak Euclidean
    return math.sqrt(np.sum((a - b) ** 2))


class KMeans:
    def __init__(self, k=3, max_iter=500, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Implementasi K-Means++
        # ----------------------------------------------------
        centroids = []
        # 1. Pilih centroid pertama secara acak
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        for _ in range(1, self.k):
            # 2. Hitung jarak kuadrat (D^2) setiap titik ke centroid terdekat
            distances_sq = np.array([min([euclidean(x, c) ** 2 for c in centroids]) for x in X])
            
            # 3. Hitung probabilitas (Probabilitas = D^2 / Sum(D^2))
            sum_distances_sq = distances_sq.sum()
            if sum_distances_sq == 0:
                 # Fallback: Jika semua jarak 0, pilih acak
                 centroids.append(X[np.random.choice(n_samples)])
            else:
                probs = distances_sq / sum_distances_sq
                # 4. Pilih titik baru berdasarkan probabilitas
                next_centroid_idx = np.random.choice(n_samples, p=probs)
                centroids.append(X[next_centroid_idx])
                 
        self.centroids = np.array(centroids)
        # ----------------------------------------------------

        for iteration in range(self.max_iter):
            # Assignment Step: Hitung label
            labels = np.array([np.argmin([euclidean(x, c) for c in self.centroids]) for x in X])

            # Update Step: Hitung centroid baru
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

            # Check Konvergensi
            if np.allclose(self.centroids, new_centroids, rtol=1e-6):
                break
            self.centroids = new_centroids

        self.labels = labels
        self.inertia = sum([euclidean(X[i], self.centroids[labels[i]]) ** 2 for i in range(n_samples)])
        return self


# ============================================
# 3️⃣ ELBOW METHOD & SILHOUETTE
# ============================================

def elbow_method(X, max_k=10):
    wcss = []
    # K-Means setidaknya butuh K=2
    # Kita akan menjalankan K-Means 3 kali untuk setiap K di Elbow Method agar lebih stabil
    n_init_elbow = 3
    
    for k in range(2, min(max_k + 1, len(X)//2 + 1)):
        min_wcss_k = float('inf')
        
        for init_run in range(n_init_elbow):
            try:
                # Menggunakan random_state=init_run untuk inisialisasi yang berbeda
                model = KMeans(k=k, random_state=init_run).fit(X)
                if model.inertia < min_wcss_k:
                    min_wcss_k = model.inertia
            except Exception:
                pass
        
        if min_wcss_k != float('inf'):
            wcss.append(min_wcss_k)
            
    return wcss


def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    n_samples = len(X)
    
    # Jika hanya ada 1 cluster atau 0 sampel, score = 0
    if len(unique_labels) < 2 or n_samples == 0:
        return 0.0

    scores = []
    
    for i in range(n_samples):
        # 1. Hitung a (Average distance to other points in the SAME cluster)
        current_label = labels[i]
        same_cluster_indices = np.where(labels == current_label)[0]
        
        # Cek jika hanya ada 1 titik di cluster, a dianggap 0 (atau skip)
        if len(same_cluster_indices) <= 1:
            continue
        
        distances_a = [euclidean(X[i], X[j]) for j in same_cluster_indices if i != j]
        a = np.mean(distances_a)

        # 2. Hitung b (Minimum average distance to points in the NEXT closest cluster)
        b_values = []
        for other_label in unique_labels:
            if other_label != current_label:
                other_cluster_indices = np.where(labels == other_label)[0]
                distances_to_other = [euclidean(X[i], X[j]) for j in other_cluster_indices]
                
                if len(distances_to_other) > 0:
                    b_values.append(np.mean(distances_to_other))
        
        # Jika tidak ada cluster lain (should not happen if len(unique_labels) >= 2)
        if not b_values:
            continue 

        b = min(b_values)
        
        # 3. Hitung Silhouette Coefficient (s)
        s = (b - a) / max(a, b)
        scores.append(s)
        
    return np.mean(scores) if scores else 0.0


# ============================================
# 4️⃣ PCA MANUAL (2D Visualization)
# ============================================

def plot_clusters(X, labels, centroids, features):
    # Jika X sudah 2D (Avg, Std), tidak perlu PCA. Langsung plot.
    if X.shape[1] == 2:
        X_pca = X
        centroids_pca = centroids
        x_label = features[0]
        y_label = features[1]
        title_suffix = "(AVG vs STD)"
    else:
        # Jika X > 2D (misal user ubah logic), fallback ke PCA manual (dari kode asli)
        X_centered = X - X.mean(axis=0)
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        principal_components = eigenvectors[:, idx[:2]]
        X_pca = X_centered.dot(principal_components)
        centroids_pca = (centroids - X.mean(axis=0)).dot(principal_components)
        x_label = "PC1"
        y_label = "PC2"
        title_suffix = "(PCA 2D)"

    plt.figure(figsize=(7, 5))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10.colors

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    s=70, alpha=0.8, edgecolors='black',
                    color=colors[i % len(colors)], label=f'Cluster {lbl}')
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c='red', marker='*', s=300, edgecolors='black', linewidths=2, label='Centroid')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Visualisasi Cluster {title_suffix}")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


# ============================================
# 5️⃣ STREAMLIT UI LOGIC
# ============================================

uploaded_file = st.file_uploader("📂 Upload file CSV kamu di sini", type=["csv"])

if uploaded_file is not None:
    # Cek apakah data sudah dimuat dan diproses di session state
    if 'df_loaded' not in st.session_state or st.session_state.df_loaded != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_loaded = uploaded_file.name
    else:
        df = st.session_state.df

    st.success(f"File berhasil dimuat! Total data: {df.shape[0]} baris, {df.shape[1]} kolom.")
    st.write("### Data Mentah (5 baris pertama)")
    st.dataframe(df.head())

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        # Pilihan Scaling
        use_zscore = st.checkbox("Gunakan Z-Score Scaling (vs. Min-Max Default)", value=False)
        scaling_method_text = "Z-Score" if use_zscore else "Min-Max"
        
        # Preprocessing
        X, features, df_display, scaling_method = load_and_preprocess(df, use_zscore)
        
        st.write(f"### Fitur Clustering (2D)")
        st.info(f"Data di-cluster HANYA menggunakan **'{features[0]}'** dan **'{features[1]}'** setelah {scaling_method} Scaling.")
        
        # Tentukan jumlah cluster
        max_k = st.slider("Pilih jumlah maksimal cluster (untuk Elbow Method)", 3, 15, 6)
        k_manual = st.number_input("Masukkan jumlah cluster (K):", min_value=2, max_value=min(10, len(X)-1), value=3)
        
        # OPSI BARU: Jumlah Inisialisasi
        n_init_manual = st.number_input(
            "Jumlah Inisialisasi K-Means (n_init):", 
            min_value=5, max_value=20, value=10, 
            help="Jumlah kali K-Means dijalankan dengan centroid awal berbeda. Pilih hasil terbaik (WCSS terendah)."
        )


    with col2:
        # Jalankan Elbow Method
        if st.button("🔍 Jalankan Elbow Method", use_container_width=True):
            with st.spinner('Menghitung WCSS untuk setiap K (3 inisialisasi per K)...'):
                wcss = elbow_method(X, max_k)
            
            if wcss:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(2, len(wcss) + 2), wcss, 'bo-', linewidth=2)
                ax.set_xlabel("Jumlah Cluster (K)")
                ax.set_ylabel("WCSS (Inertia)")
                ax.set_title("Elbow Method (Optimized)")
                st.pyplot(fig)
            else:
                st.warning("Tidak dapat menjalankan Elbow Method (Mungkin data terlalu sedikit).")

    st.markdown("---")

    if st.button("🚀 Jalankan K-Means & Hitung Silhouette Score", type="primary", use_container_width=True):
        start_time = time.time()
        
        # --- MODIFIKASI KRUSIAL: Multiple Init (n_init_manual) ---
        n_init = n_init_manual
        best_model = None
        lowest_inertia = float('inf')
        
        st.write(f"### Melakukan K-Means dengan K={k_manual} ({n_init} inisialisasi untuk stabilitas)...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(n_init):
            status_text.text(f"Iterasi Inisialisasi: {i+1}/{n_init}")
            
            # Gunakan 'i' sebagai random_state untuk inisialisasi yang berbeda
            # Kita menggunakan i + 42 untuk memastikan variasi yang lebih besar
            model = KMeans(k=k_manual, random_state=i + 42).fit(X) 
            
            # Pilih model dengan Inertia (WCSS) terkecil
            if model.inertia < lowest_inertia:
                lowest_inertia = model.inertia
                best_model = model
            
            progress_bar.progress((i + 1) / n_init)
        
        # Selesai
        progress_bar.empty()
        status_text.empty()
        # ----------------------------------------------------

        if best_model is not None:
            # Hitung Silhouette Score hanya untuk model terbaik
            silh = silhouette_score(X, best_model.labels)
            end_time = time.time()

            st.write(f"### ✅ Hasil Clustering Terbaik (K = {k_manual})")
            
            if silh > 0.5:
                st.success(f"🎉 **Silhouette Score Sukses (>0.5):** {silh:.3f}")
            else:
                st.warning(f"⚠️ **Silhouette Score:** {silh:.3f} (Belum mencapai 0.5 - Coba ubah K atau Scaling!)")

            st.info(f"Waktu Komputasi: {end_time - start_time:.2f} detik | Inertia (WCSS) Terbaik: {best_model.inertia:.2f}")

            # Tambahkan label ke DataFrame Display
            df_display["Cluster"] = [f"C{l}" for l in best_model.labels]
            
            st.write("### Data Hasil Cluster (Ditampilkan dengan Skor Asli & Fitur Tambahan)")
            st.dataframe(df_display.head(10))

            # Tampilkan Centroid (dalam data ter-normalisasi)
            st.write("#### Posisi Centroid (Data Ternormalisasi)")
            centroid_df = pd.DataFrame(best_model.centroids, columns=features)
            centroid_df.index = [f"Centroid {i}" for i in range(k_manual)]
            st.dataframe(centroid_df)

            # Visualisasi
            st.write("### Visualisasi Cluster (2D)")
            plot_clusters(X, best_model.labels, best_model.centroids, features)

        else:
            st.error("Terjadi kesalahan atau data tidak cukup untuk menjalankan K-Means.")
                
else:
    st.info("Silakan upload file CSV terlebih dahulu. Pastikan kolom skor Anda dapat diubah ke numerik.")
