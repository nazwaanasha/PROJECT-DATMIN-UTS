import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import List, Tuple, Dict

# ======================================================================================
# Konfigurasi Halaman Streamlit
# ======================================================================================
st.set_page_config(
    page_title="Segmentasi Kesiapan Riset Mahasiswa",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# BAGIAN 1: FUNGSI MANUAL (FROM SCRATCH)
# ======================================================================================
# Penjelasan: Semua fungsi di bagian ini diimplementasikan secara manual
# untuk mematuhi aturan proyek yang melarang penggunaan library clustering seperti sklearn.

# --- 1.1 Helper Function ---
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Menghitung jarak Euclidean antara dua titik (vektor NumPy).
    Jarak ini adalah metrik standar untuk mengukur 'kedekatan' antar data point.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# --- 1.2 Preprocessing Manual ---
def manual_min_max_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi data menggunakan metode Min-Max Scaling secara manual.
    Tujuannya adalah untuk mengubah skala semua fitur ke rentang [0, 1]
    agar setiap fitur memiliki bobot yang setara dalam perhitungan jarak.
    """
    df_normalized = df.copy()
    for column in df_normalized.columns:
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        range_val = max_val - min_val
        if range_val == 0:
            # Jika semua nilai dalam satu kolom sama, hasilnya menjadi 0
            df_normalized[column] = 0
        else:
            df_normalized[column] = (df_normalized[column] - min_val) / range_val
    return df_normalized

# --- 1.3 K-Means From Scratch ---
def kmeans_from_scratch(data: np.ndarray, k: int, max_iters: int = 100, random_state: int = 42) -> Tuple[np.ndarray, float]:
    """
    Implementasi algoritma K-Means dari nol.
    1. Inisialisasi: Pilih k titik data secara acak sebagai centroid awal.
    2. Assignment: Kelompokkan setiap titik data ke centroid terdekat.
    3. Update: Hitung ulang posisi centroid berdasarkan rata-rata dari semua titik dalam clusternya.
    4. Iterasi: Ulangi langkah 2-3 hingga posisi centroid tidak berubah (konvergen).
    """
    # Menjamin hasil yang sama setiap kali dijalankan dengan random_state yang sama
    np.random.seed(random_state)
    
    # 1. Inisialisasi centroid secara random dari data points
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # 2. Assignment: tentukan cluster untuk setiap data point
        clusters: List[List[int]] = [[] for _ in range(k)]
        for point_idx, point in enumerate(data):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point_idx)
        
        # Simpan centroid lama untuk cek konvergensi
        old_centroids = centroids.copy()

        # 3. Update: hitung ulang centroid baru dari rata-rata cluster
        new_centroids_list = []
        for cluster in clusters:
            if cluster: # Pastikan cluster tidak kosong
                new_centroids_list.append(np.mean(data[cluster], axis=0))
            else:
                # Jika cluster kosong, kita bisa re-inisialisasi centroid atau biarkan sama
                # Di sini, kita akan cari indeks cluster yang kosong untuk penanganan
                empty_cluster_idx = len(new_centroids_list)
                new_centroids_list.append(old_centroids[empty_cluster_idx])

        centroids = np.array(new_centroids_list)
        
        # 4. Cek konvergensi: jika centroid tidak berubah signifikan, berhenti
        if np.allclose(old_centroids, centroids):
            break

    # Buat label akhir untuk setiap data point
    labels = np.zeros(data.shape[0], dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] = i
        
    # Hitung WCSS (Within-Cluster Sum of Squares)
    wcss = sum(np.sum((data[cluster] - centroids[i])**2) for i, cluster in enumerate(clusters) if cluster)
    return labels, wcss

# --- 1.4 Hierarchical Clustering (Agglomerative) From Scratch ---
def hierarchical_from_scratch(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Implementasi Hierarchical Clustering (Agglomerative) dari nol.
    1. Inisialisasi: Setiap titik data dianggap sebagai satu cluster.
    2. Cari Pasangan Terdekat: Temukan dua cluster yang paling dekat.
       - Metode Linkage: 'Complete Linkage' digunakan, yaitu jarak diukur
         berdasarkan titik terjauh antara dua cluster.
    3. Gabung (Merge): Gabungkan dua cluster terdekat menjadi satu.
    4. Iterasi: Ulangi langkah 2-3 hingga jumlah cluster yang tersisa sesuai target (n_clusters).
    """
    n_points = data.shape[0]
    clusters = [[i] for i in range(n_points)]
    
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        merge_idx = (-1, -1)
        
        # 2. Cari dua cluster terdekat untuk digabung
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Menggunakan "complete linkage": jarak terjauh antara titik di dua cluster
                dist = max(euclidean_distance(data[p1], data[p2]) for p1 in clusters[i] for p2 in clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)
        
        # 3. Gabungkan dua cluster terdekat
        i, j = merge_idx
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    # Buat label akhir
    labels = np.zeros(n_points, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = cluster_id
    return labels

# --- 1.5 DBSCAN From Scratch ---
def dbscan_from_scratch(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implementasi DBSCAN (Density-Based Spatial Clustering of Applications with Noise) dari nol.
    1. Klasifikasi Titik: Untuk setiap titik, tentukan apakah itu 'core point', 'border point', atau 'noise'.
       - Core Point: Jika memiliki minimal 'min_samples' tetangga dalam radius 'eps'.
       - Border Point: Bukan core point, tapi berada dalam jangkauan core point.
       - Noise: Bukan keduanya.
    2. Pembentukan Cluster: Hubungkan core point yang saling terjangkau (dalam radius eps)
       untuk membentuk satu cluster. Border point akan ikut ke dalam cluster dari core point terdekatnya.
    """
    NOISE = -1
    UNVISITED = 0
    
    n_points = data.shape[0]
    labels = np.full(n_points, UNVISITED)
    cluster_id = 0

    def region_query(point_idx: int) -> List[int]:
        """Mencari semua tetangga dari sebuah titik dalam radius eps."""
        neighbors = []
        for i in range(n_points):
            if euclidean_distance(data[point_idx], data[i]) < eps:
                neighbors.append(i)
        return neighbors

    for point_idx in range(n_points):
        if labels[point_idx] != UNVISITED:
            continue
            
        neighbors = region_query(point_idx)
        
        if len(neighbors) < min_samples:
            labels[point_idx] = NOISE
        else:
            # Ini adalah 'core point', mulai cluster baru
            cluster_id += 1
            labels[point_idx] = cluster_id
            
            # Periksa semua tetangga dari core point ini
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                
                if labels[neighbor_idx] == NOISE:
                    labels[neighbor_idx] = cluster_id # Border point menjadi bagian cluster
                elif labels[neighbor_idx] == UNVISITED:
                    labels[neighbor_idx] = cluster_id
                    
                    # Jika tetangga ini juga core point, tambahkan tetangganya ke dalam antrian
                    new_neighbors = region_query(neighbor_idx)
                    if len(new_neighbors) >= min_samples:
                        neighbors.extend(new_neighbors)
                i += 1
                
    # Normalisasi ID cluster agar dimulai dari 0 (noise tetap -1)
    unique_ids = sorted([c for c in np.unique(labels) if c > 0])
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    id_map[NOISE] = -1
    labels = np.array([id_map.get(l, -1) for l in labels])
    
    return labels

# <<< TAMBAHAN BARU (1/2): Fungsi Silhouette Score >>>
def silhouette_score(data, labels):
    """Implementasi Silhouette Score secara manual."""
    n_samples = len(data)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2 or len(unique_labels) == n_samples:
        return -1 # Tidak terdefinisi jika hanya ada 1 cluster atau tiap titik adalah clusternya sendiri

    silhouette_vals = []
    for i in range(n_samples):
        current_label = labels[i]
        # Abaikan noise points (umumnya dari DBSCAN) dalam perhitungan skor
        if current_label == -1: 
            continue

        # a_i: Jarak rata-rata titik i ke titik lain dalam cluster yang sama
        same_cluster_indices = np.where(labels == current_label)[0]
        if len(same_cluster_indices) > 1:
            a_i = np.mean([euclidean_distance(data[i], data[j]) for j in same_cluster_indices if i != j])
        else:
            a_i = 0

        # b_i: Jarak rata-rata minimum titik i ke titik-titik di cluster lain
        min_avg_dist_other = float('inf')
        for label in unique_labels:
            if label == current_label or label == -1:
                continue
            
            other_cluster_indices = np.where(labels == label)[0]
            avg_dist = np.mean([euclidean_distance(data[i], data[j]) for j in other_cluster_indices])
            if avg_dist < min_avg_dist_other:
                min_avg_dist_other = avg_dist
        
        b_i = min_avg_dist_other
        
        # Hitung silhouette score untuk titik i
        if max(a_i, b_i) == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_vals.append(s_i)

    # Skor rata-rata dari semua titik adalah silhouette score total
    return np.mean(silhouette_vals) if silhouette_vals else -1

# ======================================================================================
# BAGIAN 2: FUNGSI TAMPILAN APLIKASI STREAMLIT
# ======================================================================================

def show_homepage():
    """Menampilkan halaman utama/selamat datang."""
    st.title("✍️ Aplikasi Segmentasi Kesiapan Riset Mahasiswa")
    st.markdown("---")
    st.warning("⚠️ **Penting:** Aplikasi ini dibuat sesuai aturan proyek: **Tanpa library `sklearn.cluster` atau `scipy.cluster`**. Semua fungsi preprocessing dan clustering diimplementasikan secara manual dari nol menggunakan NumPy dan Pandas.")
    
    st.markdown("""
    Selamat datang di aplikasi analisis data mahasiswa! Aplikasi ini dirancang untuk membantu Anda memahami berbagai profil kesiapan riset mahasiswa berdasarkan data survei.

    #### **Tujuan Aplikasi:**
    - **Mengelompokkan (Clustering):** Mengidentifikasi segmen-segmen mahasiswa yang memiliki karakteristik serupa dalam hal kesiapan riset.
    - **Menganalisis Profil:** Memahami ciri khas dari setiap segmen, misalnya segmen 'Siap Riset', 'Butuh Bimbingan', atau 'Belum Tertarik'.
    - **Visualisasi:** Menyajikan hasil analisis dalam bentuk yang mudah dipahami seperti Radar Chart dan Heatmap.

    #### **Cara Penggunaan:**
    1.  **Unggah Data:** Siapkan data survei Anda dalam format `.csv` atau `.xlsx`, lalu unggah menggunakan tombol di bawah.
    2.  **Pilih Metode:** Di sidebar kiri, pilih algoritma clustering yang ingin Anda gunakan (K-Means, Hierarchical, atau DBSCAN).
    3.  **Atur Parameter:** Sesuaikan parameter seperti jumlah cluster (k), epsilon (eps), dll.
    4.  **Jalankan Analisis:** Klik tombol "🚀 Jalankan Analisis" untuk memulai proses.
    5.  **Lihat Hasil:** Hasil segmentasi, profil cluster, dan **Silhouette Score** (di sidebar) akan ditampilkan.
    """)
    st.info("💡 **Tips:** Untuk K-Means, Anda bisa melihat grafik **Elbow Method** terlebih dahulu untuk mendapatkan rekomendasi jumlah cluster (k) yang optimal.")
    st.markdown("---")


def preprocess_input_data(df: pd.DataFrame) -> Tuple[pd.DataFrame | None, List[str] | None]:
    """Fungsi gabungan untuk memproses data dari file yang diunggah."""
    # Heuristik untuk memilih kolom numerik (pertanyaan survei)
    question_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not question_columns:
        st.error("❌ **Error:** Tidak dapat menemukan kolom numerik dalam data Anda. Pastikan file survei memiliki kolom dengan jawaban berupa angka (skala likert, dll).")
        return None, None
        
    data_for_clustering = df[question_columns].copy()
    
    # Penanganan Missing Values dengan Median
    if data_for_clustering.isnull().values.any():
        st.write("Mengisi nilai yang kosong (missing values) dengan median dari masing-masing kolom.")
        for col in data_for_clustering.columns:
            if data_for_clustering[col].isnull().any():
                median_val = data_for_clustering[col].median()
                data_for_clustering[col].fillna(median_val, inplace=True)
            
    # Normalisasi Data
    data_normalized = manual_min_max_scaler(data_for_clustering)
    return data_normalized, question_columns

# ========== REKOMENDASI OTOMATIS BERBASIS SILHOUETTE ==========

def recommend_k_for_kmeans(data_values: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, float, Dict[int, float]]:
    """
    Mencari k (jumlah cluster) terbaik untuk K-Means dengan mengoptimalkan silhouette score.
    Mengembalikan (best_k, best_score, scores_dict).
    """
    best_k = None
    best_score = -2.0
    scores = {}
    for k in range(k_min, k_max + 1):
        labels, _ = kmeans_from_scratch(data_values, k, random_state=42)
        s = silhouette_score(data_values, labels)
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, best_score, scores


def recommend_n_clusters_hierarchical(data_values: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, float, Dict[int, float]]:
    """
    Mencari n_clusters terbaik untuk Hierarchical (complete linkage) berdasarkan silhouette score.
    """
    best_k = None
    best_score = -2.0
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = hierarchical_from_scratch(data_values, k)
        s = silhouette_score(data_values, labels)
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, best_score, scores


def recommend_dbscan_params(data_values: np.ndarray, eps_values: List[float], min_samples_values: List[int], top_n: int = 5):
    """
    Grid search sederhana untuk DBSCAN: mencoba kombinasi eps x min_samples dan memilih
    kombinasi yang menghasilkan silhouette score tertinggi. Mengembalikan daftar top_n hasil.
    Hasil berupa list of tuples: (eps, min_samples, silhouette)
    """
    results = []
    total = len(eps_values) * len(min_samples_values)
    progress = 0
    for eps in eps_values:
        for ms in min_samples_values:
            labels = dbscan_from_scratch(data_values, eps, ms)
            s = silhouette_score(data_values, labels)
            results.append((eps, ms, s))
            progress += 1
    # Urutkan berdasarkan silhouette descending
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    return results_sorted[:top_n]

# ======================================================================================
# Bagian tampilan: plotting dan hasil (tidak mengubah fungsi-fungsi plotting sebelumnya)
# ======================================================================================

def render_sidebar(data_processed: pd.DataFrame) -> Tuple[str, Dict, bool]:
    """Menampilkan sidebar untuk input pengguna dan mengembalikan pilihan mereka."""
    with st.sidebar:
        st.header("⚙️ Pengaturan Analisis")
        algorithm = st.selectbox(
            "Pilih Metode Clustering",
            ["K-Means", "Hierarchical Clustering", "DBSCAN"],
            help="Pilih algoritma untuk mengelompokkan data."
        )

        params = {}

        # --- Rekomendasi otomatis (tombol) ditempatkan BEFORE slider sehingga hasilnya bisa menjadi default ---
        if algorithm == "K-Means":
            if st.button("🔎 Auto-rekomendasi k (maksimalkan Silhouette)"):
                with st.spinner("Mencari k terbaik berdasarkan silhouette..."):
                    best_k, best_score, scores = recommend_k_for_kmeans(data_processed.values, 2, 10)
                    st.session_state['k_reco'] = int(best_k)
                    st.success(f"Rekomendasi k: {best_k} (Silhouette = {best_score:.3f})")
                    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['silhouette']).reset_index().rename(columns={'index':'k'})
                    st.dataframe(scores_df.sort_values('silhouette', ascending=False), use_container_width=True)

            default_k = st.session_state.get('k_reco', 3)
            k = st.slider("Jumlah Cluster (k)", 2, 10, int(default_k), 1, help="Berapa banyak segmen yang ingin Anda bentuk?")
            params = {'k': k}

        elif algorithm == "Hierarchical Clustering":
            if st.button("🔎 Auto-rekomendasi n_clusters (Hierarchical)"):
                with st.spinner("Mencari n_clusters terbaik untuk hierarchical (complete linkage)..."):
                    best_k, best_score, scores = recommend_n_clusters_hierarchical(data_processed.values, 2, 10)
                    st.session_state['hier_reco'] = int(best_k)
                    st.success(f"Rekomendasi n_clusters: {best_k} (Silhouette = {best_score:.3f})")
                    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['silhouette']).reset_index().rename(columns={'index':'n_clusters'})
                    st.dataframe(scores_df.sort_values('silhouette', ascending=False), use_container_width=True)

            default_k = st.session_state.get('hier_reco', 3)
            k = st.slider("Jumlah Cluster (n_clusters)", 2, 10, int(default_k), 1, help="Berapa banyak segmen akhir yang diinginkan?")
            params = {'n_clusters': k}

        elif algorithm == "DBSCAN":
            st.markdown("_Catatan: rekomendasi DBSCAN memerlukan pencarian grid kecil — mungkin butuh beberapa detik tergantung ukuran data._")
            # Tombol rekomendasi DBSCAN
            if st.button("🔎 Auto-rekomendasi eps & min_samples (DBSCAN)"):
                with st.spinner("Mencari kombinasi eps & min_samples terbaik untuk DBSCAN..."):
                    # Definisi default grid (cukup kecil agar cepat)
                    eps_vals = list(np.linspace(0.05, 1.5, 15))
                    min_samples_vals = list(range(2, 11))
                    top_results = recommend_dbscan_params(data_processed.values, eps_vals, min_samples_vals, top_n=10)
                    # Simpan rekomendasi tertinggi
                    if top_results:
                        best_eps, best_ms, best_score = top_results[0]
                        st.session_state['dbscan_reco_eps'] = float(best_eps)
                        st.session_state['dbscan_reco_min_samples'] = int(best_ms)
                        st.success(f"Rekomendasi DBSCAN: eps={best_eps:.3f}, min_samples={best_ms} (Silhouette={best_score:.3f})")
                        top_df = pd.DataFrame(top_results, columns=['eps', 'min_samples', 'silhouette'])
                        st.dataframe(top_df, use_container_width=True)
                    else:
                        st.info("Tidak menemukan kombinasi yang valid (mungkin semua percobaan menghasilkan 0 atau 1 cluster).")

            eps_suggestion = st.session_state.get('dbscan_reco_eps', 0.5)
            eps = st.slider("Epsilon (eps)", 0.01, 2.0, float(eps_suggestion), 0.01, help="Jarak maksimum antar titik untuk dianggap sebagai tetangga.")
            min_samples = st.session_state.get('dbscan_reco_min_samples', 5)
            min_samples = st.slider("Minimum Samples", 2, 20, int(min_samples), 1, help="Jumlah minimum titik dalam sebuah lingkungan agar menjadi 'core point'.")
            params = {'eps': eps, 'min_samples': min_samples}

        st.markdown("---")
        run_button = st.button("🚀 Jalankan Analisis", use_container_width=True, type="primary")

    # Tampilkan Elbow Method di main area jika K-Means dipilih
    if algorithm == "K-Means" and data_processed is not None:
        plot_elbow_method(data_processed)

    return algorithm, params, run_button


def plot_elbow_method(data: pd.DataFrame):
    """Menampilkan plot Elbow Method untuk K-Means yang diimplementasikan manual."""
    with st.expander("📈 Lihat Rekomendasi Jumlah Cluster (Elbow Method untuk K-Means)"):
        st.info("""
        Grafik ini membantu menentukan jumlah cluster ("k") yang paling pas. Carilah titik "siku" (elbow) di mana penurunan grafik mulai melandai secara signifikan. 
        Titik tersebut adalah kandidat kuat untuk jumlah cluster optimal, karena penambahan cluster setelahnya tidak lagi memberikan peningkatan kualitas pemisahan yang berarti.
        """)
        wcss_values = []
        k_range = range(2, 11)
        
        progress_bar = st.progress(0, text="Menghitung WCSS untuk berbagai nilai k...")
        # Caching agar tidak dihitung ulang setiap saat
        @st.cache_data
        def calculate_wcss_for_range(_data_values, _k_range):
            wcss_list = []
            for i, k in enumerate(_k_range):
                _, wcss = kmeans_from_scratch(_data_values, k, random_state=42)
                wcss_list.append(wcss)
                # Progress bar di luar cache
            return wcss_list

        wcss_values = calculate_wcss_for_range(data.values, k_range)
        progress_bar.progress(1.0, text="Perhitungan WCSS selesai.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, wcss_values, marker='o', linestyle='--', color='b')
        ax.set_title('Elbow Method (Implementasi Manual)', fontsize=16)
        ax.set_xlabel('Jumlah Cluster (k)', fontsize=12)
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax.grid(True)
        st.pyplot(fig)
        


def plot_radar_charts(cluster_profiles: pd.DataFrame):
    """Membuat dan menampilkan Radar Chart untuk setiap profil cluster."""
    labels = cluster_profiles.columns.tolist()
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Tutup poligon

    num_clusters = len(cluster_profiles)
    num_cols = min(3, num_clusters)
    num_rows = math.ceil(num_clusters / num_cols)

    fig, axes = plt.subplots(figsize=(5 * num_cols, 5 * num_rows), nrows=num_rows, ncols=num_cols, subplot_kw=dict(polar=True))
    axes = axes.flatten() if num_clusters > 1 else [axes]

    for i, (cluster_id, row) in enumerate(cluster_profiles.iterrows()):
        ax = axes[i]
        values = row.tolist()
        values += values[:1] # Tutup poligon
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Segmen {cluster_id}")
        ax.fill(angles, values, alpha=0.25)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(f"Profil Segmen {cluster_id}", pad=20, weight='bold', size='large')
        ax.grid(True)

    # Sembunyikan subplot yang tidak terpakai
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    
def plot_cluster_scatter(data: pd.DataFrame, labels: np.ndarray):
    """
    Visualisasi khas clustering (2D) menggunakan reduksi dimensi sederhana.
    Menggunakan PCA manual (tanpa sklearn) untuk memproyeksikan data ke 2D.
    """
    # Reduksi dimensi manual (PCA sederhana)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    
    # Ambil 2 komponen utama dengan eigenvalue terbesar
    idx = np.argsort(eig_vals)[::-1]
    principal_components = eig_vecs[:, idx[:2]]
    reduced_data = np.dot(data_centered, principal_components)
    
    # Buat DataFrame hasil proyeksi
    df_vis = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    df_vis["Cluster"] = labels
    
    # Plot visualisasi
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="PC1", y="PC2", hue="Cluster", data=df_vis,
        palette="tab10", s=60, edgecolor="black"
    )
    ax.set_title("Visualisasi 2D Hasil Clustering (PCA Manual)", fontsize=14)
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    ax.grid(True)
    st.pyplot(fig)


def render_results(df_original: pd.DataFrame, data_processed: pd.DataFrame, labels: np.ndarray, question_columns: List[str]):
    """Menampilkan semua hasil analisis dan visualisasi dalam format tab."""
    df_result = df_original.copy()
    df_result['Segmentasi'] = labels
    # Ganti label -1 (noise) menjadi teks yang lebih deskriptif
    df_result['Segmentasi'] = df_result['Segmentasi'].apply(lambda x: 'Noise/Outlier' if x == -1 else f'Segmen {x}')

    num_clusters = len([l for l in np.unique(labels) if l != -1])
    num_noise = np.sum(labels == -1)

    st.success(f"✅ Analisis selesai! Ditemukan **{num_clusters} segmen** dan **{num_noise} outlier** (jika menggunakan DBSCAN).")
    
    data_with_labels = data_processed.copy()
    data_with_labels['Segmentasi'] = labels
    
    # Filter out noise/outliers for profile analysis
    data_for_profiling = data_with_labels[data_with_labels['Segmentasi'] != -1]
    
    if not data_for_profiling.empty:
        cluster_profiles = data_for_profiling.groupby('Segmentasi').mean()
        
        tab1, tab2, tab3 = st.tabs(["📊 **Profil Segmen**", "📄 **Interpretasi (Contoh)**", "📋 **Data Hasil**"])

        with tab1:
            st.header("Visualisasi Profil Setiap Segmen")
            st.markdown("Visualisasi ini menunjukkan karakteristik rata-rata dari setiap segmen yang terbentuk. Gunakan ini untuk memahami apa yang membedakan satu segmen dari yang lain.")
            
            st.subheader("🕸️ Radar Chart")
            st.write("Bandingkan bentuk 'jaring laba-laba' antar segmen. Segmen dengan jaring yang lebih lebar pada area tertentu menunjukkan skor yang lebih tinggi pada dimensi tersebut.")
            plot_radar_charts(cluster_profiles)

            st.subheader("🔥 Heatmap")
            st.write("Lihat perbedaan skor secara keseluruhan. Warna yang lebih terang menunjukkan skor rata-rata yang lebih tinggi untuk segmen tersebut pada pertanyaan terkait.")
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(16, 8))
            sns.heatmap(cluster_profiles.T, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax_heatmap, linewidths=.5)
            ax_heatmap.set_title('Rangkuman Rata-rata Skor per Segmen', fontsize=16)
            st.pyplot(fig_heatmap)
            
            st.subheader("🌈 Visualisasi 2D Clustering")
            st.write("Visualisasi ini menunjukkan bagaimana setiap data terkelompok dalam ruang dua dimensi hasil proyeksi PCA sederhana.")
            plot_cluster_scatter(data_processed, labels)

        with tab2:
            st.header("Bagaimana Menginterpretasikan Hasil Ini?")
            st.markdown("""
            Interpretasi adalah kunci dari analisis clustering. Tujuannya adalah memberikan **nama** dan **narasi** yang bermakna untuk setiap segmen. Berikut adalah contoh template untuk membantu Anda:
            """)
            for i, profile in cluster_profiles.iterrows():
                st.subheader(f"Analisis untuk Segmen {i}")
                # Cari 3 skor tertinggi dan 3 terendah
                top_features = profile.nlargest(3).index.tolist()
                bottom_features = profile.nsmallest(3).index.tolist()
                st.markdown(f"""
                - **Karakteristik Menonjol (Skor Tertinggi):** Mahasiswa di segmen ini menunjukkan skor tinggi pada: `{', '.join(top_features)}`.
                - **Area Pengembangan (Skor Terendah):** Mereka cenderung memiliki skor lebih rendah pada: `{', '.join(bottom_features)}`.
                - **Kesimpulan (Hipotesis):** Berdasarkan profil ini, kita bisa memberikan nama pada segmen ini sebagai **'Nama Segmen (Contoh: Siap Tempur Mandiri)'**. Mereka adalah kelompok yang... [lanjutkan narasi Anda, misal: '...sudah memiliki pemahaman metodologi yang baik tetapi kurang percaya diri dalam presentasi'].
                - **Rekomendasi Tindakan:** Untuk segmen ini, intervensi yang mungkin cocok adalah... [contoh: '...pelatihan public speaking atau sesi konsultasi untuk memvalidasi ide riset mereka.'].
                """)
            st.warning("**Catatan:** Interpretasi di atas adalah contoh. Sesuaikan narasi dan nama segmen dengan konteks pertanyaan survei dan hasil analisis Anda.")

        with tab3:
            st.header("Data Hasil Segmentasi per Responden")
            st.dataframe(df_result, use_container_width=True)
            
            # Opsi untuk mengunduh hasil
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(df_result)
            st.download_button(
                label="📥 Unduh Hasil (.csv)",
                data=csv,
                file_name="hasil_segmentasi_mahasiswa.csv",
                mime="text/csv",
            )
    else:
        st.warning("Tidak ada cluster yang terbentuk (semua data dianggap noise). Coba sesuaikan parameter `eps` dan `min_samples` pada DBSCAN.")

# ======================================================================================
# BAGIAN 3: FUNGSI UTAMA APLIKASI
# ======================================================================================
def main():
    show_homepage()
    
    uploaded_file = st.file_uploader(
        "Langkah 1: Unggah file data survei Anda",
        type=["csv", "xlsx"]
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("✔️ File berhasil diunggah!")
            
            st.subheader("Pratinjau Data Asli (5 Baris Pertama)")
            st.dataframe(df.head())
            
            # Lakukan preprocessing
            data_processed, question_columns = preprocess_input_data(df)
            
            if data_processed is not None:
                st.markdown("---")
                st.header("Langkah 2: Pilih Metode & Jalankan Analisis")
                
                algorithm, params, run_button = render_sidebar(data_processed)
                
                if run_button:
                    st.markdown("---")
                    st.header("Langkah 3: Hasil Analisis")
                    with st.spinner(f"⏳ Menjalankan {algorithm} (implementasi manual)... Proses ini mungkin butuh waktu beberapa saat."):
                        labels = None
                        data_values = data_processed.values # Gunakan NumPy array untuk performa
                        
                        if algorithm == "K-Means":
                            labels, _ = kmeans_from_scratch(data_values, **params, random_state=42)
                        elif algorithm == "Hierarchical Clustering":
                            labels = hierarchical_from_scratch(data_values, **params)
                        elif algorithm == "DBSCAN":
                            labels = dbscan_from_scratch(data_values, **params)
                        
                        # <<< TAMBAHAN BARU (2/2): Hitung dan tampilkan Silhouette Score di sidebar >>>
                        if labels is not None:
                            score = silhouette_score(data_values, labels)
                            st.sidebar.metric(
                                label="Kualitas Cluster (Silhouette)",
                                value=f"{score:.3f}",
                                help="Skor antara -1 (buruk) dan 1 (sangat baik). Semakin tinggi skor, semakin baik segmen Anda terbentuk."
                            )
                            
                            # Tambahan interpretasi nilai silhouette
                            st.sidebar.markdown("### 📊 Interpretasi Nilai Silhouette")
                            st.sidebar.markdown("""
                            | Nilai Silhouette | Interpretasi |
                            |:----------------:|:-------------|
                            | **+1.0** | Clustering **sangat baik** — setiap titik data sangat cocok dengan clusternya sendiri dan sangat jauh dari cluster lain. |
                            | **0.7 – 1.0** | **Sangat bagus** — cluster jelas terpisah dengan baik. |
                            | **0.5 – 0.7** | **Baik** — pemisahan antar-cluster cukup jelas, hasil clustering bisa diterima. |
                            | **0.25 – 0.5** | **Sedang** — ada tumpang tindih antar-cluster, mungkin perlu optimasi jumlah cluster atau metode lain. |
                            | **0.0 – 0.25** | **Kurang baik** — banyak data salah tempat atau cluster kurang terpisah. |
                            | **< 0** | **Buruk** — data cenderung ditempatkan di cluster yang salah. |
                            """)

                        render_results(df, data_processed, labels, question_columns)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    main()
