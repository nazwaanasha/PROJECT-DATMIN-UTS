import streamlit as st   # Library untuk membuat aplikasi web interaktif
import pandas as pd      # Library untuk membaca dan mengolah data tabular (CSV, Excel, dll)
import numpy as np       # Library untuk perhitungan numerik/matriks
import matplotlib.pyplot as plt  # Library untuk membuat visualisasi grafik
import random            # Library untuk randomisasi (dipakai untuk inisialisasi centroid K-Means)

# ==============================
# Fungsi K-Means Manual
# ==============================

# Fungsi untuk menghitung jarak Euclidean antar 2 titik
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))  # Rumus √(∑(xi - yi)^2)

# Implementasi algoritma K-Means secara manual
def kmeans_manual(X, k=3, max_iter=100):
    # random.seed(42)  
    # np.random.seed(42)

    # Inisialisasi centroid 
    centroids = X[random.sample(range(len(X)), k)]

    # Looping maksimal sebanyak max_iter
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]  # List kosong untuk menyimpan anggota tiap cluster

        # Assign tiap titik ke cluster terdekat
        for idx, point in enumerate(X):
            distances = [euclidean_distance(point, c) for c in centroids]  # Hitung jarak ke semua centroid
            cluster_idx = np.argmin(distances)  # Cari cluster dengan jarak terpendek
            clusters[cluster_idx].append(idx)   # Masukkan index data ke cluster tersebut

        old_centroids = centroids.copy()  # Simpan centroid lama sebelum update

        # Update centroid → rata-rata titik dalam cluster
        for i in range(k):
            if clusters[i]:  # Jika cluster tidak kosong
                centroids[i] = np.mean(X[clusters[i]], axis=0)

        # Jika centroid sudah konvergen (tidak berubah), hentikan loop
        if np.allclose(old_centroids, centroids):
            break

    # Buat label untuk tiap data sesuai cluster-nya
    labels = np.zeros(len(X))
    for cluster_idx, cluster_points in enumerate(clusters):
        for point_idx in cluster_points:
            labels[point_idx] = cluster_idx

    return labels, centroids  # Kembalikan hasil clustering (label dan posisi centroid)


# ==============================
# Streamlit App
# ==============================
st.title("📊 Clustered Research Readiness for Students")  
# Judul aplikasi Streamlit

st.write("Aplikasi untuk melakukan clustering kesiapan riset mahasiswa menggunakan **K-Means manual**.")
# Deskripsi aplikasi

# Upload file CSV berisi hasil kuesioner
uploaded_file = st.file_uploader("Upload file CSV kuesioner", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Baca file CSV ke DataFrame

    st.subheader("Data Awal")
    st.dataframe(df.head())  # Tampilkan 5 data awal

    # ==============================
    # Preprocessing
    # ==============================
    st.subheader("Preprocessing Data")

    # Kolom yang dibuang karena tidak dipakai untuk clustering
    drop_cols = ["Timestamp", "Nama", "Usia", "Program Studi", "Saran"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode Jenis Kelamin: Laki-laki=0, Perempuan=1
    if "Jenis Kelamin" in df.columns:
        df["Jenis Kelamin"] = df["Jenis Kelamin"].map({"Laki-laki": 0, "Perempuan": 1})

    # Ambil angka semester dari teks ("Semester 3" → 3)
    if "Semester" in df.columns:
        df["Semester"] = df["Semester"].astype(str).str.extract(r"(\d+)").astype(int)

    # Ambil hanya kolom numerik untuk clustering
    df_numeric = df.select_dtypes(include=[np.number])

    # Simpan daftar kolom numerik (Likert + gender + semester)
    kolom_kesiapan = df_numeric.columns.tolist()

    # Standarisasi data (mean=0, std=1) agar skala setara
    X = df_numeric.values.astype(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    st.write("✅ Data berhasil diproses. Siap untuk clustering.")
    st.dataframe(df_numeric.head())

    # ==============================
    # K-Means Clustering
    # ==============================
    if st.button("Jalankan Clustering"):
        # Jalankan K-Means manual
        labels, centroids = kmeans_manual(X, k=3)
        df["Cluster"] = labels.astype(int)  # Tambahkan hasil cluster ke DataFrame

        st.subheader("Hasil Clustering")
        st.dataframe(df)  # Tampilkan tabel hasil clustering

        # Ringkasan cluster
        st.subheader("📌 Ringkasan Cluster")

        st.write("Jumlah anggota tiap cluster:")
        st.write(df["Cluster"].value_counts())  # Hitung jumlah data per cluster

        st.write("Rata-rata readiness per cluster:")
        st.write(df.groupby("Cluster")[kolom_kesiapan].mean())  # Rata-rata tiap variabel per cluster

        # Distribusi gender per cluster
        if "Jenis Kelamin" in df.columns:
            st.write("Distribusi Gender per cluster:")
            st.write(pd.crosstab(df["Cluster"], df["Jenis Kelamin"]))

        # Distribusi semester per cluster
        if "Semester" in df.columns:
            st.write("Distribusi Semester per cluster:")
            st.write(pd.crosstab(df["Cluster"], df["Semester"]))

        # ==============================
        # Visualisasi
        # ==============================
        st.subheader("📊 Visualisasi")
        
        from sklearn.decomposition import PCA  # PCA untuk reduksi dimensi

        # Visualisasi cluster dalam 2D menggunakan PCA
        st.write("Visualisasi cluster dalam 2D (PCA)")

        # Reduksi data ke 2 dimensi (PC1 & PC2)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Proyeksi centroid ke ruang PCA
        centroids_pca = pca.transform(centroids)

        # Buat scatter plot
        fig, ax = plt.subplots(figsize=(7,5))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Warna khas KMeans/Orange (biru, oranye, hijau)

        # Plot data per cluster
        for cluster_id in np.unique(labels):
            cluster_points = X_pca[labels == cluster_id]
            ax.scatter(cluster_points[:,0], cluster_points[:,1], 
                    s=50, c=colors[int(cluster_id)], label=f"Cluster {int(cluster_id)}", alpha=0.7)

        # Plot centroid
        ax.scatter(centroids_pca[:,0], centroids_pca[:,1], 
                s=200, c="red", marker="X", edgecolors="black", linewidths=2, label="Centroid")

        # Tambahkan detail plot
        ax.set_title("K-Means Clustering (2D PCA Projection)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)  # Tampilkan plot di Streamlit

        # Pie chart jumlah mahasiswa per cluster
        fig1, ax1 = plt.subplots()
        df["Cluster"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        # Bar chart rata-rata readiness per cluster
        fig2, ax2 = plt.subplots(figsize=(8,5))
        df.groupby("Cluster")[kolom_kesiapan].mean().T.plot(kind="bar", ax=ax2)
        ax2.set_title("Rata-rata Skor Readiness per Cluster")
        st.pyplot(fig2)

        # ==============================
        # Interpretasi hasil cluster
        # ==============================
        st.subheader("📝 Interpretasi")
        st.write("""
        - Cluster 0: **Kesiapan rendah**  
        - Cluster 1: **Kesiapan sedang**  
        - Cluster 2: **Kesiapan tinggi**  
        """)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Clustered Research Readiness for Students",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2563eb;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
    }
    </style>
""", unsafe_allow_html=True)
