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

# Title
st.markdown('<div class="main-title">🎓 Clustered Research Readiness for Students</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analisis Kesiapan Riset Mahasiswa Menggunakan K-Means Clustering</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Konfigurasi")
    num_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=5, value=3)
    st.markdown("---")
    st.markdown("### 📖 Tentang Aplikasi")
    st.info("""
    Aplikasi ini menganalisis kesiapan riset mahasiswa menggunakan metode K-Means (implementasi manual).
    
    **Fitur:**
    - Upload data kuesioner CSV
    - Preprocessing otomatis
    - Clustering mahasiswa
    - Visualisasi interaktif
    - Interpretasi hasil
    """)

# --- Functions ---
def preprocess_data(df):
    """Preprocessing data kuesioner:
       - Drop kolom teks yang tidak diperlukan untuk clustering
       - Encode gender
       - Extract semester number (raw regex string)
    """
    # Keep a copy of original for analyses that need Program Studi / Nama
    df_clean = df.copy()

    # Drop columns that shouldn't be used as numeric features
    columns_to_drop = ['Timestamp', 'Nama', 'Usia', 'Saran']  # keep Program Studi in original df
    df_clean = df_clean.drop(columns=[c for c in columns_to_drop if c in df_clean.columns], errors='ignore')

    # Encode Jenis Kelamin (0 = laki-laki, 1 = perempuan). If missing, try to fillna with 0.
    if 'Jenis Kelamin' in df_clean.columns:
        df_clean['Jenis Kelamin'] = df_clean['Jenis Kelamin'].fillna('').apply(
            lambda x: 0 if 'laki' in str(x).lower() else (1 if 'perempuan' in str(x).lower() or 'wanita' in str(x).lower() else np.nan)
        )

    # Extract semester number safely (raw string to avoid escape warnings)
    if 'Semester' in df_clean.columns:
        df_clean['Semester'] = df_clean['Semester'].astype(str).str.extract(r'(\d+)')
        df_clean['Semester'] = pd.to_numeric(df_clean['Semester'], errors='coerce').fillna(-1).astype(int)

    # Convert Likert columns to numeric where possible
    # Attempt to coerce remaining columns to numeric (Q1..Qn)
    for col in df_clean.columns:
        if col not in ['Jenis Kelamin', 'Semester', 'Program Studi'] :
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')

    return df_clean

def normalize_data(X):
    """Z-score normalization with small epsilon to avoid divide-by-zero."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    eps = 1e-8
    X_norm = (X - mean) / (std + eps)
    return X_norm

def manual_kmeans(X, k=3, max_iters=100, random_state=42):
    """Manual K-Means without sklearn.
       Handles empty clusters by reinitializing centroid to a random point.
    """
    np.random.seed(random_state)
    n = X.shape[0]
    k = min(k, n)  # avoid k > n
    random_idx = np.random.choice(n, k, replace=False)
    centroids = X[random_idx].astype(float)

    for _ in range(max_iters):
        # compute pairwise distances (n x k)
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)
            else:
                # reinitialize empty centroid to a random sample
                new_centroids[i] = X[np.random.choice(n)]

        # convergence check
        if np.allclose(centroids, new_centroids, atol=1e-4, equal_nan=True):
            break
        centroids = new_centroids

    return labels, centroids

def perform_clustering(df, n_clusters=3):
    """Perform clustering on numeric columns (including Jenis Kelamin & Semester)."""
    # Select numeric columns only from processed df
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    if df_numeric.shape[1] == 0:
        raise ValueError("Tidak ada kolom numerik yang ditemukan untuk clustering. Pastikan Q1..Qn berupa angka (1-5).")

    # Keep feature column names for interpretation (exclude 'Jenis Kelamin' and 'Semester' if desired)
    feature_columns = [col for col in df_numeric.columns if col not in ['Jenis Kelamin', 'Semester']]

    # Normalize entire numeric array (so gender & semester also contribute, but scaled)
    X = df_numeric.values
    X_scaled = normalize_data(X)

    # Run manual kmeans
    clusters, centroids = manual_kmeans(X_scaled, k=n_clusters)
    return clusters, feature_columns, df_numeric.columns.tolist()

def calculate_cluster_characteristics(df_processed, clusters, feature_columns):
    """Calculate cluster sizes, avg readiness (on feature_columns), gender & semester distributions."""
    df_with_clusters = df_processed.copy()
    df_with_clusters['Cluster'] = clusters

    characteristics = []
    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        count = len(cluster_data)
        percentage = (count / len(df_with_clusters)) * 100

        # Average scores: use only feature_columns if present, otherwise mean over numeric columns
        if feature_columns:
            avg_scores = cluster_data[feature_columns].apply(pd.to_numeric, errors='coerce').mean()
            avg_readiness = np.nanmean(avg_scores.values)
        else:
            avg_readiness = cluster_data.select_dtypes(include=[np.number]).mean().mean()

        # Gender distribution (guard if column missing)
        male_count = female_count = 0
        if 'Jenis Kelamin' in cluster_data.columns:
            gender_dist = cluster_data['Jenis Kelamin'].value_counts(dropna=True)
            male_count = int(gender_dist.get(0, 0))
            female_count = int(gender_dist.get(1, 0))

        semester_dist = {}
        if 'Semester' in cluster_data.columns:
            semester_dist = cluster_data['Semester'].value_counts().to_dict()

        if avg_readiness >= 4.0:
            interpretation = "🟢 Tingkat Kesiapan Tinggi - Mahasiswa sangat siap melakukan riset"
            level = "Tinggi"
        elif avg_readiness >= 3.0:
            interpretation = "🟡 Tingkat Kesiapan Sedang - Mahasiswa cukup siap dengan beberapa area yang perlu diperbaiki"
            level = "Sedang"
        else:
            interpretation = "🔴 Tingkat Kesiapan Rendah - Mahasiswa memerlukan pelatihan intensif sebelum memulai riset"
            level = "Rendah"

        characteristics.append({
            'Cluster': cluster_id,
            'Jumlah': count,
            'Persentase': percentage,
            'Avg Readiness': float(avg_readiness) if not np.isnan(avg_readiness) else 0.0,
            'Level': level,
            'Laki-laki': male_count,
            'Perempuan': female_count,
            'Semester': semester_dist,
            'Interpretasi': interpretation
        })

    # sort by avg readiness descending
    characteristics.sort(key=lambda x: x['Avg Readiness'], reverse=True)
    return df_with_clusters, characteristics

# --- Main App ---
uploaded_file = st.file_uploader("📁 Upload File CSV Kuesioner", type=['csv'])

if uploaded_file is not None:
    try:
        # Read CSV (attempt common encodings)
        df_original = pd.read_csv(uploaded_file)
        st.success(f"✅ Data berhasil diupload! Total: {len(df_original)} mahasiswa")

        with st.expander("👀 Lihat Data Asli"):
            st.dataframe(df_original.head(10))

        # Preprocess (this returns a df used for clustering; original kept for Program Studi display)
        df_processed = preprocess_data(df_original)

        # Run clustering
        clusters, feature_columns, numeric_cols = perform_clustering(df_processed, n_clusters=num_clusters)

        # Calculate characteristics
        df_with_clusters, characteristics = calculate_cluster_characteristics(df_processed, clusters, feature_columns)

        st.markdown("---")
        st.header("📊 Visualisasi Hasil Clustering")

        # PIE
        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
        fig_pie = px.pie(values=cluster_counts.values,
                         names=[f'Cluster {i}' for i in cluster_counts.index],
                         title='Distribusi Mahasiswa per Cluster',
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

        # BAR Avg Readiness
        readiness_data = pd.DataFrame([
            {'Cluster': f"Cluster {c['Cluster']}", 'Avg Readiness': c['Avg Readiness']}
            for c in characteristics
        ])
        fig_bar = px.bar(readiness_data, x='Cluster', y='Avg Readiness',
                         title='Rata-rata Kesiapan Riset per Cluster',
                         color='Avg Readiness', color_continuous_scale='blues')
        fig_bar.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_bar, use_container_width=True)

        # Karakteristik cluster
        st.header("🎯 Karakteristik Setiap Cluster")
        for char in characteristics:
            with st.expander(f"📌 Cluster {char['Cluster']} - Kesiapan {char['Level']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Jumlah Mahasiswa", char['Jumlah'])
                with col2:
                    st.metric("Persentase", f"{char['Persentase']:.1f}%")
                with col3:
                    st.metric("Rata-rata Skor", f"{char['Avg Readiness']:.2f}")
                with col4:
                    st.metric("Gender Ratio", f"♂️{char['Laki-laki']} : ♀️{char['Perempuan']}")
                st.info(char['Interpretasi'])
                # Distribusi semester (readable)
                if char['Semester']:
                    semester_text = ", ".join([f"Sem {k}: {v}" for k, v in sorted(char['Semester'].items())])
                    st.markdown("**Distribusi Semester:**")
                    st.write(semester_text)

        # Tabel hasil
        st.header("📋 Tabel Hasil Clustering")
        df_display = df_original.copy()
        df_display['Cluster'] = clusters
        cols = ['Cluster'] + [col for col in df_display.columns if col != 'Cluster']
        df_display = df_display[cols]
        st.dataframe(df_display, use_container_width=True)

        # Download
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Clustering (CSV)", data=csv,
                           file_name="hasil_clustering.csv", mime="text/csv")

        # === ANALISIS LANJUTAN ===
        st.markdown("---")
        st.header("📌 Analisis Lanjutan & Insight")

        # Gabungkan kembali data asli + cluster (gunakan original karena punya Program Studi)
        df_analysis = df_original.copy()
        df_analysis['Cluster'] = clusters

        # Program Studi vs Cluster (proportion)
        if 'Program Studi' in df_analysis.columns:
            st.subheader("📖 Program Studi vs Cluster (Proporsi per Program Studi)")
            prodi_cluster = (df_analysis
                             .groupby('Program Studi')['Cluster']
                             .value_counts(normalize=True)
                             .unstack(fill_value=0)
                             .sort_index(axis=0))
            # show as percentage
            st.dataframe(prodi_cluster.style.format("{:.1%}"))

            # heatmap
            try:
                fig_heatmap = px.imshow(prodi_cluster,
                                       labels=dict(x="Cluster", y="Program Studi", color="Proporsi"),
                                       aspect="auto",
                                       color_continuous_scale="Blues")
                fig_heatmap.update_traces(texttemplate="%{z:.0%}", textfont={"size":12})
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception:
                # fallback: show simple table only
                pass

            st.markdown("💡 **Interpretasi:** Program studi dengan persentase tinggi pada cluster readiness rendah menunjukkan kebutuhan prioritas pelatihan.")

        # Semester vs Cluster (proporsi per semester)
        if 'Semester' in df_analysis.columns:
            st.subheader("🎓 Semester vs Cluster (Proporsi per Semester)")
            semester_cluster = (df_analysis
                                .groupby('Semester')['Cluster']
                                .value_counts(normalize=True)
                                .unstack(fill_value=0)
                                .sort_index(axis=0))
            st.dataframe(semester_cluster.style.format("{:.1%}"))

            # heatmap semester
            try:
                fig_sem_heat = px.imshow(semester_cluster,
                                        labels=dict(x="Cluster", y="Semester", color="Proporsi"),
                                        aspect="auto",
                                        color_continuous_scale="Blues")
                fig_sem_heat.update_traces(texttemplate="%{z:.0%}", textfont={"size":12})
                st.plotly_chart(fig_sem_heat, use_container_width=True)
            except Exception:
                pass

            st.markdown("💡 **Interpretasi:** Biasanya semester awal akan menunjukkan proporsi lebih besar pada cluster readiness rendah; gunakan insight ini untuk menargetkan pelatihan dasar.")

        # Rekomendasi praktis per cluster
        st.subheader("📝 Rekomendasi Berdasarkan Cluster")
        for char in characteristics:
            if char['Level'] == "Rendah":
                rekom = "⚠️ **Butuh pelatihan intensif:** fokus pada metodologi riset dasar & analisis data."
            elif char['Level'] == "Sedang":
                rekom = "📚 **Perlu penguatan tambahan:** misalnya workshop penulisan akademik atau praktik analisis data."
            else:
                rekom = "✅ **Siap riset:** arahkan ke proyek penelitian, asistensi riset, atau publikasi."
            st.markdown(f"- **Cluster {char['Cluster']}** ({char['Level']}, {char['Jumlah']} mahasiswa): {rekom}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Pastikan format CSV sesuai dengan template yang diminta (kolom: Timestamp, Nama, Usia, Jenis Kelamin, Semester, Program Studi, Q1..Qn, Saran).")

else:
    st.info("👆 Silakan upload file CSV kuesioner untuk memulai analisis")

    st.markdown("""
    ### 📋 Format CSV yang Dibutuhkan:
    
    File CSV harus memiliki kolom berikut:
    - **Timestamp** (opsional)
    - **Nama** (opsional)
    - **Usia** (opsional)
    - **Jenis Kelamin**: Laki-laki / Perempuan
    - **Semester**: format mengandung angka (contoh: "Semester 3" atau "3")
    - **Program Studi**
    - **Q1..Q15**: Pertanyaan Likert (skala 1-5) — kolom ini harus berupa angka
    - **Saran** (opsional)
    """)