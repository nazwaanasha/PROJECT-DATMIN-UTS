import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Konfigurasi Streamlit
st.set_page_config(page_title="K-Means Manual Upgrade", layout="wide")
st.title("🔬 K-Means Manual Clustering (Upgrade Fitur)")
st.write("Dioptimasi untuk mendapatkan Silhouette Score tinggi ($>0.5$) melalui **Outlier Removal** dan **Reduksi Dimensi (PCA)**.")

# ============================================
# 1️⃣ PREPROCESSING & FEATURE REDUCTION (Dengan Outlier Removal dan PCA)
# ============================================

def outlier_removal_iqr(df_features):
    """
    Menghapus outliers dari DataFrame menggunakan metode Interquartile Range (IQR).
    Hanya beroperasi pada kolom numerik.
    """
    df_cleaned = df_features.copy()
    initial_count = len(df_cleaned)
    mask = pd.Series(True, index=df_cleaned.index)

    for col in df_cleaned.columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Masking: Tahan True untuk data yang BUKAN outlier
        col_mask = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
        mask = mask & col_mask # Kombinasikan mask dari semua kolom

    df_cleaned = df_cleaned[mask]
    removed_count = initial_count - len(df_cleaned)
    return df_cleaned, removed_count, mask

def load_and_preprocess(df, use_zscore=False, remove_outliers=False, use_multidim=False, n_components_pca=0):
    """
    Memuat, membersihkan, mengisi NaN, melakukan rekayasa fitur (AVG, STD), 
    mengelola outlier, menormalisasi, dan opsional melakukan reduksi dimensi PCA.
    """
    # --- Step 1 & 2: Clean, Impute, and Feature Engineering ---
    skip_keywords = ['timestamp', 'nama', 'usia', 'jenis', 'semester', 'program', 'saran']
    relevant_cols = [c for c in df.columns if not any(k in c.lower() for k in skip_keywords)]
    
    df_clean_scores = df[relevant_cols].apply(pd.to_numeric, errors='coerce')

    for col in df_clean_scores.columns:
        df_clean_scores[col].fillna(df_clean_scores[col].median(), inplace=True)

    # Tambahkan fitur rekayasa (tetap hitung meskipun tidak dipakai)
    df_clean_scores['avg_score'] = df_clean_scores[relevant_cols].mean(axis=1)
    df_clean_scores['std_score'] = df_clean_scores[relevant_cols].std(axis=1)

    # --- Feature Selection ---
    if use_multidim:
        # Gunakan SEMUA kolom skor (relevant_cols)
        df_features = df_clean_scores[relevant_cols]
    else:
        # Gunakan HANYA fitur rekayasa (AVG dan STD Score)
        df_features = df_clean_scores[['avg_score', 'std_score']]
    
    # --- Step 3: Outlier Removal ---
    outlier_count = 0
    if remove_outliers:
        df_features_cleaned, outlier_count, mask = outlier_removal_iqr(df_features)
        df_features = df_features_cleaned
    
    # Sinkronisasi df_display
    df_display = df.copy()
    for col in ['avg_score', 'std_score']:
        df_display[col] = df_clean_scores[col]
    
    if remove_outliers:
        df_display = df_display[mask].reset_index(drop=True)
        
    # --- Step 4: Normalization (WAJIB sebelum PCA) ---
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

    df_final_features = df_norm.copy()
    
    # --- NEW STEP: PCA for Feature Reduction (if requested) ---
    if use_multidim and n_components_pca > 0 and len(df_final_features) > df_final_features.shape[1] and df_final_features.shape[1] > n_components_pca:
        
        X_scaled = df_final_features.values
        X_centered = X_scaled - X_scaled.mean(axis=0)
        
        try:
            cov_matrix = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Sort komponen berdasarkan Eigenvalues
            idx = eigenvalues.argsort()[::-1]
            principal_components = eigenvectors[:, idx[:n_components_pca]]
            
            # Transformasi data
            X_pca = X_centered.dot(principal_components)
            
            # Update DataFrame fitur final
            df_final_features = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components_pca)], index=df_final_features.index)
            
            # Update nama metode scaling
            scaling_method = f"{scaling_method} $\\to$ PCA ({n_components_pca} Komponen)"
            
        except np.linalg.LinAlgError:
            # Jika PCA gagal, gunakan fitur yang sudah dinormalisasi (df_norm)
            st.warning("PCA gagal: Matriks kovarians singular. Melanjutkan dengan fitur asli yang dinormalisasi.")
    
    return df_final_features.values, df_final_features.columns.tolist(), df_display, scaling_method, outlier_count


# ============================================
# 2️⃣ K-MEANS MANUAL IMPLEMENTATION (Tetap, sudah K-Means++)
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
        
        # Cek jika data terlalu sedikit
        if n_samples < 2:
            raise ValueError("Data terlalu sedikit untuk clustering.")
        
        # Implementasi K-Means++
        centroids = []
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        for _ in range(1, self.k):
            # Hitung jarak kuadrat minimum ke centroid terdekat (D(x)^2)
            distances_sq = np.array([min([euclidean(x, c) ** 2 for c in centroids]) for x in X])
            
            sum_distances_sq = distances_sq.sum()
            
            if sum_distances_sq == 0:
                 # Jika semua jarak 0, berhenti atau pilih secara acak dari yang tersisa
                 candidate_indices = [i for i in range(n_samples) if not np.any([np.allclose(X[i], c) for c in centroids])]
                 if candidate_indices:
                     centroids.append(X[np.random.choice(candidate_indices)])
                 else:
                     break
            else:
                probs = distances_sq / sum_distances_sq
                
                # FIX KRITIKAL: Memastikan titik yang sudah menjadi centroid memiliki probabilitas 0.
                mask_selected = np.zeros(n_samples, dtype=bool)
                for c in centroids:
                    matches_c = np.allclose(X, c, atol=1e-8)
                    mask_selected = mask_selected | matches_c # Gabungkan mask
                
                probs[mask_selected] = 0
                
                sum_probs = probs.sum()
                if sum_probs > 0:
                    probs /= sum_probs
                    next_centroid_idx = np.random.choice(n_samples, p=probs)
                    centroids.append(X[next_centroid_idx])
                else:
                    break
                    
        if len(centroids) < self.k:
            self.k = len(centroids) 

        self.centroids = np.array(centroids)
        
        # Loop K-Means utama
        for iteration in range(self.max_iter):
            # Assignment Step
            distances = np.array([[euclidean(x, c) for c in self.centroids] for x in X])
            labels = np.argmin(distances, axis=1)

            # Update Step
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[labels == i]
                if cluster_points.size > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[i]) # Jaga centroid lama jika cluster kosong

            new_centroids = np.array(new_centroids)
            
            # Check Konvergensi
            if np.allclose(self.centroids, new_centroids, rtol=1e-6):
                break
            self.centroids = new_centroids

        self.labels = labels
        self.inertia = sum([euclidean(X[i], self.centroids[labels[i]]) ** 2 for i in range(n_samples)])
        return self


# ============================================
# 3️⃣ ELBOW METHOD & SILHOUETTE (Tetap)
# ============================================

def elbow_method(X, max_k=10):
    wcss = []
    n_init_elbow = 3
    
    # Batasi K maksimal
    max_k_limit = min(max_k + 1, len(X)//2 + 1)
    
    for k in range(2, max_k_limit):
        min_wcss_k = float('inf')
        
        for init_run in range(n_init_elbow):
            try:
                # Menggunakan random_state yang lebih unik
                model = KMeans(k=k, random_state=init_run + 100).fit(X) 
                if model.inertia < min_wcss_k:
                    min_wcss_k = model.inertia
            except Exception as e:
                pass
        
        if min_wcss_k != float('inf'):
            wcss.append(min_wcss_k)
            
    return wcss

def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    n_samples = len(X)
    
    if len(unique_labels) < 2 or n_samples == 0:
        return 0.0

    scores = []
    
    for i in range(n_samples):
        current_label = labels[i]
        same_cluster_indices = np.where(labels == current_label)[0]
        
        if len(same_cluster_indices) <= 1:
            continue
        
        distances_a = [euclidean(X[i], X[j]) for j in same_cluster_indices if i != j]
        a = np.mean(distances_a)

        b_values = []
        for other_label in unique_labels:
            if other_label != current_label:
                other_cluster_indices = np.where(labels == other_label)[0]
                distances_to_other = [euclidean(X[i], X[j]) for j in other_cluster_indices]
                
                if len(distances_to_other) > 0:
                    b_values.append(np.mean(distances_to_other))
        
        if not b_values:
            continue 

        b = min(b_values)
        
        s = (b - a) / max(a, b)
        scores.append(s)
        
    return np.mean(scores) if scores else 0.0


# ============================================
# 4️⃣ PCA MANUAL & VISUALISASI
# ============================================

def plot_clusters(X, labels, centroids, features):
    # Jika X sudah 2D (Avg, Std), tidak perlu PCA. Langsung plot.
    if X.shape[1] == 2:
        X_pca = X
        centroids_pca = centroids
        x_label = features[0]
        y_label = features[1]
        title_suffix = "(AVG vs STD)"
    elif features[0].startswith('PC'):
        # Data sudah dalam komponen PCA (digunakan untuk clustering). Plot 2 komponen pertama.
        X_pca = X[:, :2]
        centroids_pca = centroids[:, :2]
        x_label = features[0]
        y_label = features[1]
        title_suffix = f"({X.shape[1]} Komponen PCA)"
    else:
        # Fallback ke PCA manual jika dimensi > 2 (Raw features used for clustering)
        X_centered = X - X.mean(axis=0)
        
        if X.shape[0] <= X.shape[1]:
            st.warning("Data terlalu sedikit (jumlah baris $\le$ jumlah fitur) untuk menjalankan PCA. Plot tidak dapat ditampilkan.")
            return

        cov_matrix = np.cov(X_centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        except np.linalg.LinAlgError:
            st.warning("Matriks kovarians singular, tidak dapat menghitung Eigenvalues. Plot tidak dapat ditampilkan.")
            return

        idx = eigenvalues.argsort()[::-1]
        principal_components = eigenvectors[:, idx[:2]]
        
        if principal_components.shape[1] < 2:
             st.warning("Tidak cukup variansi untuk menghasilkan 2 komponen utama. Plot tidak dapat ditampilkan.")
             return
             
        X_pca = X_centered.dot(principal_components)
        centroids_pca = (centroids - X.mean(axis=0)).dot(principal_components)
        x_label = "PC1"
        y_label = "PC2"
        title_suffix = f"({X.shape[1]} Fitur ke 2D via PCA)"

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

def plot_cluster_distribution(labels):
    """Membuat bar chart untuk melihat jumlah anggota di setiap cluster."""
    plt.figure(figsize=(7, 4))
    
    # Hitung jumlah anggota per cluster
    unique, counts = np.unique(labels, return_counts=True)
    
    # Buat Bar Chart
    colors = plt.cm.tab10.colors
    plt.bar([f'Cluster {i}' for i in unique], counts, color=[colors[i % len(colors)] for i in unique])
    
    # Tambahkan nilai di atas bar
    max_count = max(counts) if counts.size > 0 else 1
    for i, count in enumerate(counts):
        plt.text(i, count + max_count * 0.02, str(count), ha='center', fontsize=10)
        
    plt.xlabel("Cluster ID")
    plt.ylabel("Jumlah Anggota")
    plt.title("Distribusi Anggota Per Cluster")
    plt.grid(axis='y', linestyle='--')
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

    st.success(f"File berhasil dimuat! Total data awal: {df.shape[0]} baris, {df.shape[1]} kolom.")
    st.write("### Data Mentah (5 baris pertama)")
    st.dataframe(df.head())

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### Opsi Preprocessing & Parameter")
        
        # Pilihan Outlier Removal
        remove_outliers = st.checkbox("✅ Hapus Outlier (IQR 1.5) sebelum Scaling", value=True, 
                                      help="Sangat direkomendasikan untuk K-Means agar tidak sensitif terhadap data ekstrem.")
        
        # Pilihan Scaling
        use_zscore = st.checkbox("Gunakan Z-Score Scaling (vs. Min-Max Default)", value=False)
        
        # Pilihan Multidimensi
        use_multidim = st.checkbox("➕ Gunakan Semua Kolom Skor (Multi-Dimensi)", value=False,
                                   help="Menggunakan semua kolom skor asli untuk clustering.")
        
        n_components_pca = 0
        if use_multidim:
            st.markdown("---")
            st.markdown("#### Opsi Reduksi Dimensi (untuk Multi-Dimensi)")
            use_pca_for_clustering = st.checkbox("⬇️ Gunakan PCA untuk Fitur Clustering", value=False,
                                                 help="SANGAT DISARANKAN jika skor rendah (0.1 - 0.4). Coba 3 atau 4 komponen.")
            
            if use_pca_for_clustering:
                # Pastikan batas komponen PCA
                max_pca_comp = min(10, df.shape[1] - 7) # Kurangi kolom non-score
                if max_pca_comp < 2: max_pca_comp = 2
                    
                n_components_pca = st.slider(
                    "Pilih Jumlah Komponen PCA (Fitur Clustering)",
                    min_value=2,
                    max_value=max_pca_comp,
                    value=min(3, max_pca_comp)
                )

        # Preprocessing (UPDATED CALL)
        X, features, df_display, scaling_method, outlier_count = load_and_preprocess(df, use_zscore, remove_outliers, use_multidim, n_components_pca)
        
        # UPDATED INFO TEXT
        if use_multidim and n_components_pca > 0:
            feature_text = f"PCA-Reduced: {n_components_pca} Komponen"
        elif use_multidim:
             feature_text = f"{len(features)} Fitur Asli"
        else:
             feature_text = "2 Fitur (AVG & STD)"
        
        st.info(f"Fitur Clustering: **{feature_text}** ({scaling_method} Scaling).")

        if remove_outliers:
            st.warning(f"Jumlah Outlier Dihapus: **{outlier_count}** data. Data tersisa: **{len(X)}**.")
        
        # Tentukan jumlah cluster
        max_k = st.slider("Pilih jumlah maksimal cluster (untuk Elbow Method)", 3, 15, 6)
        k_manual = st.number_input("Masukkan jumlah cluster (K):", min_value=2, max_value=min(10, len(X)-1), value=3)
        
        n_init_manual = st.number_input(
            "Jumlah Inisialisasi K-Means (n_init):", 
            min_value=5, max_value=20, value=15, 
            help="Jumlah kali K-Means dijalankan dengan centroid awal berbeda. Pilih hasil terbaik (WCSS terendah)."
        )


    with col2:
        st.write("### Elbow Method (Penentuan K)")
        # Jalankan Elbow Method
        if st.button("🔍 Jalankan Elbow Method", use_container_width=True):
            if len(X) < 4:
                st.error("Data terlalu sedikit setelah Preprocessing untuk menjalankan Elbow Method.")
            else:
                with st.spinner('Menghitung WCSS untuk setiap K (3 inisialisasi per K)...'):
                    wcss = elbow_method(X, max_k)
                
                if wcss:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(range(2, len(wcss) + 2), wcss, 'bo-', linewidth=2)
                    ax.set_xlabel("Jumlah Cluster (K)")
                    ax.set_ylabel("WCSS (Inertia)")
                    ax.set_title("Elbow Method (Optimized)")
                    st.pyplot(fig)
                    
                    # --- PANDUAN BARU SETELAH ELBOW DIHITUNG ---
                    st.markdown("""
                    #### 💡 Panduan Memilih K Terbaik
                    Perhatikan grafik di atas. Titik **'siku'** (*elbow*) adalah tempat penurunan WCSS (Inertia) mulai mendatar.
                    Angka $K$ pada titik tersebut adalah nilai optimal untuk klaster Anda.

                    1. **Analisis Plot:** Jika siku berada di $K=4$ atau $K=5$, segera masukkan nilai tersebut ke dalam kotak **"Masukkan jumlah cluster (K):"** di kolom kiri.
                    2. **Uji Coba Scaling/PCA:** Jika skor masih rendah, coba centang/non-aktifkan **'Gunakan Z-Score Scaling'** dan **'Gunakan PCA'**.
                    3. **Jalankan:** Klik tombol **"🚀 Jalankan K-Means & Hitung Silhouette Score"** lagi!
                    """)
                    # --- END PANDUAN BARU ---

                else:
                    st.warning("Tidak dapat menjalankan Elbow Method (Mungkin data terlalu sedikit).")

    st.markdown("---")

    if st.button("🚀 Jalankan K-Means & Hitung Silhouette Score", type="primary", use_container_width=True):
        if len(X) < 4:
            st.error("Data terlalu sedikit setelah Preprocessing. Coba non-aktifkan Outlier Removal atau tambahkan lebih banyak data.")
        elif k_manual < 2 or k_manual >= len(X):
            st.error(f"Nilai K={k_manual} tidak valid untuk {len(X)} data.")
        else:
            start_time = time.time()
            
            # --- MODIFIKASI KRUSIAL: Multiple Init ---
            n_init = n_init_manual
            best_model = None
            lowest_inertia = float('inf')
            
            st.write(f"### Melakukan K-Means dengan K={k_manual} ({n_init} inisialisasi untuk stabilitas)...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(n_init):
                status_text.text(f"Iterasi Inisialisasi: {i+1}/{n_init}")
                
                # Gunakan random_state yang lebih variatif
                model = KMeans(k=k_manual, random_state=i * 13 + 7).fit(X) 
                
                # Pilih model dengan Inertia (WCSS) terkecil
                if model.inertia < lowest_inertia:
                    lowest_inertia = model.inertia
                    best_model = model
                
                progress_bar.progress((i + 1) / n_init)
            
            progress_bar.empty()
            status_text.empty()
            # ----------------------------------------------------

            if best_model is not None:
                # Hitung Silhouette Score hanya untuk model terbaik
                silh = silhouette_score(X, best_model.labels)
                end_time = time.time()

                st.write(f"### ✅ Hasil Clustering Terbaik (K = {k_manual})")
                
                if silh > 0.5:
                    st.balloons()
                    st.success(f"🎉 **Silhouette Score Sukses (>0.5):** {silh:.3f}")
                else:
                    st.warning(f"⚠️ **Silhouette Score:** {silh:.3f} (Belum mencapai 0.5 - Coba ubah K, gunakan Outlier Removal, atau ganti Scaling/PCA!)")

                st.info(f"Waktu Komputasi: {end_time - start_time:.2f} detik | Inertia (WCSS) Terbaik: {best_model.inertia:.2f}")

                # Tambahkan label ke DataFrame Display
                df_display["Cluster"] = [f"C{l}" for l in best_model.labels]
                
                st.write("### Data Hasil Cluster (Top 10)")
                st.dataframe(df_display.head(10))

                # Tampilkan Centroid (dalam data ter-normalisasi)
                st.write("#### Posisi Centroid (Data Ternormalisasi)")
                centroid_df = pd.DataFrame(best_model.centroids, columns=features)
                centroid_df.index = [f"Centroid {i}" for i in range(best_model.k)]
                st.dataframe(centroid_df)

                # Visualisasi
                st.write("### Visualisasi Cluster (2D)")
                plot_clusters(X, best_model.labels, best_model.centroids, features)
                
                # NEW FEATURE: Distribusi Cluster
                st.write("### Distribusi Anggota Cluster")
                plot_cluster_distribution(best_model.labels)

            else:
                st.error("Terjadi kesalahan atau data tidak cukup untuk menjalankan K-Means.")
                
else:
    st.info("Silakan upload file CSV terlebih dahulu. Pastikan kolom skor Anda dapat diubah ke numerik.")
