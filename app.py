# ======================================================
# File: app.py
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
import preprocessing as pre
import evaluation as eva
import utils as utl

# =====================================================================================
# PAGE CONFIGURATION
# =====================================================================================
st.set_page_config(
    page_title="Segmentasi Clustering Literasi Riset",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =====================================================================================
# MAIN APPLICATION
# =====================================================================================

def main():
    st.title("🎓 Segmentasi Kesiapan Riset Mahasiswa FMIPA Unpad")
    st.markdown("""<div class="markdown-box">Aplikasi clustering komprehensif dengan preprocessing detail, feature engineering, auto-recommendation, dan interpretasi otomatis berbasis data survei literasi riset.</div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Panel Kontrol")
    uploaded_file = st.sidebar.file_uploader("1. Unggah Data Survei (.csv)", type=["csv"])

    if uploaded_file is not None:
        df_original = pd.read_csv(uploaded_file)
        
        st.header("1️⃣ Pratinjau Data Asli")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Responden", df_original.shape[0])
        col2.metric("Total Kolom", df_original.shape[1])
        col3.metric("Missing Values", df_original.isnull().sum().sum())
        
        st.dataframe(df_original.head(10))

        # Preprocessing Controls
        st.sidebar.header("2. Opsi Preprocessing")
        
        with st.sidebar.expander("Pembersihan & Outlier", expanded=True):
            missing_numeric = st.selectbox("Metode Missing Value (Numerik)", ['mean', 'median'])
            missing_categorical = st.selectbox("Metode Missing Value (Kategorik)", ['mode', 'constant'])
            handle_outliers_flag = st.checkbox("Tangani Outlier (Winsorize)", value=True)
            outlier_limit = st.slider("Batas Quantile Outlier", 0.0, 0.1, 0.01, step=0.005, format="%.3f")

        with st.sidebar.expander("Transformasi & Reduksi Dimensi", expanded=True):
            scaling_method = st.selectbox("Metode Scaling Fitur", ['standard', 'minmax'])
            use_pca = st.checkbox("Gunakan PCA untuk Reduksi Dimensi", value=True)
            pca_components = st.number_input("Jumlah Komponen PCA", min_value=2, max_value=20, value=2)

        # Run Preprocessing
        if st.sidebar.button("🚀 Jalankan Preprocessing"):
            with st.spinner("Melakukan preprocessing komprehensif..."):
                report = {}
                df_processed = df_original.copy()
                
                # Step 1: Clean names
                df_processed = pre.clean_column_names(df_processed)
                st.session_state['column_mapping'] = {old: new for old, new in zip(df_original.columns, df_processed.columns)}
                
                # Step 2: Statistics before
                numeric_cols_initial = df_processed.select_dtypes(include=np.number).columns
                stats_before = pre.get_statistics_summary(df_processed, numeric_cols_initial)
                st.session_state['stats_before'] = stats_before
                
                # Step 3: Handle Duplicates
                n_duplicates = df_processed.duplicated().sum()
                if n_duplicates > 0:
                    df_processed.drop_duplicates(inplace=True, ignore_index=True)
                report['Duplicates'] = {"removed": n_duplicates}

                # Step 4: Handle Missing Values
                df_processed, missing_report = pre.handle_missing_values(df_processed, missing_numeric, missing_categorical)
                report['Missing Values'] = missing_report

                # Step 5: Handle Outliers
                outlier_affected_before = None
                if handle_outliers_flag:
                    numeric_cols = df_processed.select_dtypes(include=np.number).columns
                    df_processed, outlier_report, outlier_affected_before = pre.handle_outliers_winsorize(
                        df_processed, numeric_cols, outlier_limit, 1-outlier_limit
                    )
                    report['Outliers'] = outlier_report
                    st.session_state['outlier_before_data'] = outlier_affected_before
                
                # Step 6: Encoding
                df_processed, encoding_report = pre.encode_categorical_features(df_processed)
                report['Encoding'] = encoding_report

                # Step 7: Scaling
                df_processed, scaling_report, scaling_before = pre.scale_numeric_features(df_processed, scaling_method)
                report['Scaling'] = scaling_report
                st.session_state['scaling_before_data'] = scaling_before
                
                # Step 8: Feature Engineering
                df_processed = pre.feature_engineering_advanced(df_processed)
                fe_cols = ['mean_skor', 'std_skor', 'prop_tinggi', 'prop_rendah', 'skor_stabilitas', 
                          'gap_tinggi_rendah', 's_ready', 'profil_aktif']
                available_fe = [c for c in fe_cols if c in df_processed.columns]
                report['Feature Engineering'] = {"created_features": available_fe}
                
                # Statistics after
                numeric_cols_final = df_processed.select_dtypes(include=np.number).columns
                stats_after = pre.get_statistics_summary(df_processed, numeric_cols_final)
                st.session_state['stats_after'] = stats_after
                
                st.session_state['processed_data_full'] = df_processed
                st.session_state['report'] = report
                st.session_state['original_data'] = df_original

                # Step 9: PCA
                if use_pca:
                    data_values = df_processed.values
                    X_pca, eigenvalues, variance_explained = pre.perform_pca(data_values, n_components=pca_components)
                    report['PCA'] = {"used": True, "components": pca_components, 
                                    "variance_explained": variance_explained, "eigenvalues": eigenvalues}
                    st.session_state['final_data_for_clustering'] = X_pca
                else:
                    report['PCA'] = {"used": False}
                    st.session_state['final_data_for_clustering'] = df_processed.values
                
                st.session_state['preprocessing_done'] = True
        
        # Display Preprocessing Results
        if 'preprocessing_done' in st.session_state and st.session_state['preprocessing_done']:
            st.header("2️⃣ Hasil Preprocessing Detail")
            
            report = st.session_state.get('report', {})
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Data Duplikat", report.get('Duplicates', {}).get('removed', 0))
            col2.metric("Missing Values Filled", report.get('Missing Values', {}).get('total_filled', 0))
            
            outlier_count = sum([v['capped_count'] for v in report.get('Outliers', {}).values()])
            col3.metric("Outliers Handled", outlier_count)
            
            fe_count = len(report.get('Feature Engineering', {}).get('created_features', []))
            col4.metric("Features Created", fe_count)
            
            # Detailed Reports
            with st.expander("📋 Laporan Detail Preprocessing"):
                st.json(report)
            
            # Outlier Visualization
            if 'outlier_before_data' in st.session_state and report.get('Outliers'):
                st.subheader("📉 Analisis Outlier: Distribusi Sebelum & Sesudah Winsorizing")
                
                outlier_report = report['Outliers']
                affected_cols = list(outlier_report.keys())[:3]  # Show top 3
                
                if affected_cols:
                    for col in affected_cols:
                        col_report = outlier_report[col]
                        if col_report['capped_pct'] > 5:
                            st.warning(f"⚠️ **{col}**: {col_report['capped_pct']:.1f}% data terpengaruh (>{5}%)")
                        else:
                            st.info(f"✅ **{col}**: {col_report['capped_pct']:.1f}% data terpengaruh")
                    
                    fig_outlier = utl.plot_distribution_comparison(
                        st.session_state['outlier_before_data'],
                        st.session_state['processed_data_full'],
                        affected_cols
                    )
                    st.plotly_chart(fig_outlier, use_container_width=True)
            
            # Statistics Comparison 
            with st.expander("📊 Statistik Kolom: Sebelum & Sesudah"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sebelum Preprocessing:**")
                    stats_before_df = st.session_state.get('stats_before', pd.DataFrame())
                    if not stats_before_df.empty:
                        # Format only numeric columns
                        formatted_df = stats_before_df.copy()
                        for col in ['Mean', 'Std', 'Min', 'Max', 'Missing %']:
                            if col in formatted_df.columns:
                                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                        st.dataframe(formatted_df)
                with col2:
                    st.write("**Sesudah Preprocessing:**")
                    stats_after_df = st.session_state.get('stats_after', pd.DataFrame())
                    if not stats_after_df.empty:
                        # Format only numeric columns
                        formatted_df = stats_after_df.copy()
                        for col in ['Mean', 'Std', 'Min', 'Max', 'Missing %']:
                            if col in formatted_df.columns:
                                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                        st.dataframe(formatted_df)
        
        # Auto-Recommendation
            st.sidebar.header("3. Auto-Rekomendasi Parameter")
            
            if st.sidebar.button("🔍 Cari Parameter Optimal"):
                data_for_clustering = st.session_state['final_data_for_clustering']
                
                st.header("3️⃣ Rekomendasi Parameter Optimal")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("K-Means: Optimal K")
                    with st.spinner("Menghitung..."):
                        k_results = utl.auto_recommend_k(data_for_clustering)
                        st.success(f"**K Optimal: {k_results['optimal_k']}**")
                        st.plotly_chart(utl.plot_elbow_method(k_results['k_values'], 
                                                         k_results['wcss_scores'], 
                                                         k_results['silhouette_scores']), 
                                      use_container_width=True)
                        st.session_state['recommended_k'] = k_results['optimal_k']
                
                with col2:
                    st.subheader("DBSCAN: Optimal Params")
                    with st.spinner("Menghitung..."):
                        best_params = utl.auto_recommend_eps(data_for_clustering)
                        if best_params['eps'] is not None:
                            st.success(f"**Eps: {best_params['eps']:.2f}**")
                            st.success(f"**Min Samples: {best_params['min_samples']}**")
                            st.info(f"Silhouette: {best_params['silhouette']:.3f}")
                            st.session_state['recommended_eps'] = best_params['eps']
                            st.session_state['recommended_min_samples'] = best_params['min_samples']
                        else:
                            st.warning("Parameter optimal tidak ditemukan")
                
                with col3:
                    st.subheader("Hierarchical: Optimal K")
                    with st.spinner("Menghitung..."):
                        h_results = utl.auto_recommend_k_hierarchical(data_for_clustering)
                        st.success(f"**K Optimal: {h_results['optimal_k']}**")
                        k_vals = [k for k, _ in h_results['scores']]
                        sil_vals = [s for _, s in h_results['scores']]
                        fig_h = go.Figure(data=go.Scatter(x=k_vals, y=sil_vals, mode='lines+markers',
                                                         line=dict(color='#2563EB', width=3)))
                        fig_h.update_layout(title='Silhouette vs K', xaxis_title='K', 
                                          yaxis_title='Silhouette', template='plotly_white', height=300)
                        st.plotly_chart(fig_h, use_container_width=True)
                        st.session_state['recommended_k_hierarchical'] = h_results['optimal_k']

            # Clustering
            st.sidebar.header("4. Opsi Clustering")
            algorithm = st.sidebar.selectbox("Pilih Algoritma", ["K-Means", "Hierarchical (Divisive)", "DBSCAN"])
            
            params = {}
            if algorithm == "K-Means":
                default_k = st.session_state.get('recommended_k', 3)
                params['k'] = st.sidebar.slider("Jumlah Cluster (k)", 2, 10, default_k)
            elif algorithm == "Hierarchical (Divisive)":
                default_k_h = st.session_state.get('recommended_k_hierarchical', 3)
                params['n_clusters'] = st.sidebar.slider("Jumlah Cluster", 2, 10, default_k_h)
            elif algorithm == "DBSCAN":
                default_eps = st.session_state.get('recommended_eps', 1.5)
                default_min = st.session_state.get('recommended_min_samples', 5)
                params['eps'] = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, float(default_eps), step=0.1)
                params['min_samples'] = st.sidebar.slider("Min Samples", 2, 20, int(default_min))

            # Run Clustering
            if st.sidebar.button("🧠 Jalankan Clustering"):
                st.header("4️⃣ Hasil Clustering & Evaluasi Komprehensif")
                
                with st.spinner(f"Menjalankan {algorithm}..."):
                    data_for_clustering = st.session_state['final_data_for_clustering']
                    labels = None
                    cluster_history = None

                    if algorithm == "K-Means":
                        labels, centroids = clu.kmeans_from_scratch(data_for_clustering, **params, random_state=42)
                        st.session_state['centroids'] = centroids
                    elif algorithm == "Hierarchical (Divisive)":
                        labels, cluster_history = clu.divisive_hierarchical_from_scratch(data_for_clustering, **params)
                        st.session_state['cluster_history'] = cluster_history
                    else:
                        labels = clu.dbscan_from_scratch(data_for_clustering, **params)
                    
                    st.session_state['labels'] = labels
                    st.session_state['algorithm'] = algorithm
                    
                    # Get gender for entropy
                    df_original_indexed = st.session_state['original_data'].copy()
                    gender_col = df_original_indexed.iloc[:, 3].values if df_original_indexed.shape[1] > 3 else np.array(['Unknown']*len(labels))
                    
                    # Evaluation Metrics
                    metrics = {
                        "Silhouette Score": eva.silhouette_score_from_scratch(data_for_clustering, labels),
                        "Calinski-Harabasz Index": eva.calinski_harabasz_score_from_scratch(data_for_clustering, labels),
                        "Davies-Bouldin Index": eva.davies_bouldin_score_from_scratch(data_for_clustering, labels),
                        "Dunn Index": eva.dunn_index_from_scratch(data_for_clustering, labels),
                        "Compactness Ratio": eva.compactness_ratio_from_scratch(data_for_clustering, labels),
                        "Cluster Balance Ratio": eva.cluster_balance_ratio(labels),
                        "Entropy Index (Gender)": eva.entropy_index(labels, gender_col)
                    }
                    
                    st.session_state['metrics'] = metrics
                    
                    # Display Metrics
                    st.subheader("📈 Metrik Evaluasi Clustering")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
                    col2.metric("Dunn Index", f"{metrics['Dunn Index']:.4f}")
                    col3.metric("Balance Ratio", f"{metrics['Cluster Balance Ratio']:.4f}")
                    col4.metric("Entropy (Gender)", f"{metrics['Entropy Index (Gender)']:.4f}")
                    
                    # Format metrics DataFrame
                    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
                    metrics_display = metrics_df.copy()
                    metrics_display['Score'] = metrics_display['Score'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(metrics_display)
                    
                    with st.expander("ℹ️ Interpretasi Metrik"):
                        st.markdown("""
                        **Metrik Utama:**
                        - **Silhouette Score** [-1, 1]: Mendekati +1 = cluster terpisah baik. >0.5 bagus, >0.7 sangat baik.
                        - **Calinski-Harabasz Index** [0, ∞): Semakin tinggi semakin baik. Rasio separasi antar vs dalam cluster.
                        - **Davies-Bouldin Index** [0, ∞): Mendekati 0 = cluster kompak dan terpisah.
                        - **Dunn Index** [0, ∞): Semakin tinggi semakin baik. Rasio jarak minimum antar-cluster vs diameter maksimum.
                        - **Compactness Ratio** [0, 1]: Mendekati 0 = cluster sangat kompak.
                        
                        **Metrik Tambahan:**
                        - **Cluster Balance Ratio** [0, 1]: Mendekati 1 = ukuran cluster seimbang.
                        - **Entropy Index**: Mengukur keragaman distribusi gender dalam cluster. Rendah = homogen.
                        """)
                    
                    # Enhanced Interpretation
                    st.subheader("🎯 Interpretasi Otomatis Cluster")
                    df_fe = st.session_state['processed_data_full']
                    interpretation_df = utl.interpret_clusters_enhanced(data_for_clustering, labels, df_fe, st.session_state['original_data'])
                    st.dataframe(interpretation_df, use_container_width=True)
                    st.session_state['interpretation'] = interpretation_df
                    
                    # Aspect Summary per Cluster
                    st.subheader("📊 Ringkasan Aspek per Cluster")
                    aspect_summary = utl.get_cluster_aspect_summary(df_fe, labels)
                    # Format aspect summary
                    aspect_display = aspect_summary.copy()
                    for col in aspect_display.columns[1:]:
                        aspect_display[col] = aspect_display[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(aspect_display)
                    
                    # Visualization
                    st.subheader("📊 Visualisasi Hasil Clustering")
                    
                    if algorithm == "K-Means":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.session_state.get('report', {}).get('PCA', {}).get('used'):
                                st.plotly_chart(utl.plot_clusters_2d(data_for_clustering, labels), use_container_width=True)
                        with col2:
                            k_results = utl.auto_recommend_k(data_for_clustering, range(2, 11))
                            st.plotly_chart(utl.plot_elbow_method(k_results['k_values'], 
                                                             k_results['wcss_scores'], 
                                                             k_results['silhouette_scores']), 
                                          use_container_width=True)
                    
                    elif algorithm == "Hierarchical (Divisive)":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.session_state.get('report', {}).get('PCA', {}).get('used'):
                                st.plotly_chart(utl.plot_clusters_2d(data_for_clustering, labels), use_container_width=True)
                        with col2:
                            if cluster_history:
                                st.plotly_chart(utl.plot_dendrogram_simple(cluster_history, data_for_clustering.shape[0]), 
                                              use_container_width=True)
                    
                    else:  # DBSCAN
                        n_noise = np.sum(labels == -1)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Jumlah Cluster", n_clusters)
                        col2.metric("Noise Points", n_noise)
                        col3.metric("% Noise", f"{(n_noise/len(labels)*100):.1f}%")
                        
                        if st.session_state.get('report', {}).get('PCA', {}).get('used'):
                            st.plotly_chart(utl.plot_dbscan_results(data_for_clustering, labels), use_container_width=True)
                    
                    # Insights & Recommendations
                    st.subheader("💡 Insight & Rekomendasi Akademik")
                    
                    # Find best and worst clusters
                    cluster_means = interpretation_df.copy()
                    cluster_means['mean_numeric'] = cluster_means['Mean Skor'].str.replace(',', '.').astype(float)
                    best_cluster = cluster_means.loc[cluster_means['mean_numeric'].idxmax()]
                    worst_cluster = cluster_means.loc[cluster_means['mean_numeric'].idxmin()]
                    
                    # Best cluster insight
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>🏆 Cluster Terbaik: {best_cluster['Cluster']}</h4>
                    <p><strong>{best_cluster['Interpretasi']}</strong> - {best_cluster['Deskripsi']}</p>
                    <p>📊 <strong>{best_cluster['Jumlah Mahasiswa']}</strong> mahasiswa ({best_cluster['Persentase']}) dengan mean skor <strong>{best_cluster['Mean Skor']}</strong></p>
                    <p>🎯 <strong>Rekomendasi:</strong> {best_cluster['Rekomendasi']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Worst cluster insight
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>⚠️ Cluster yang Perlu Perhatian: {worst_cluster['Cluster']}</h4>
                    <p><strong>{worst_cluster['Interpretasi']}</strong> - {worst_cluster['Deskripsi']}</p>
                    <p>📊 <strong>{worst_cluster['Jumlah Mahasiswa']}</strong> mahasiswa ({worst_cluster['Persentase']}) dengan mean skor <strong>{worst_cluster['Mean Skor']}</strong></p>
                    <p>🎯 <strong>Rekomendasi:</strong> {worst_cluster['Rekomendasi']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Overall insights
                    avg_silhouette = metrics['Silhouette Score']
                    if avg_silhouette > 0.5:
                        quality_msg = "✅ Kualitas Clustering: BAIK - Cluster terbentuk dengan baik dan terpisah jelas."
                        quality_class = "success-box"
                    elif avg_silhouette > 0.25:
                        quality_msg = "⚠️ Kualitas Clustering: CUKUP - Ada overlapping antar cluster, pertimbangkan penyesuaian parameter."
                        quality_class = "warning-box"
                    else:
                        quality_msg = "❌ Kualitas Clustering: KURANG - Cluster kurang terpisah, coba algoritma atau parameter lain."
                        quality_class = "warning-box"
                    
                    st.markdown(f'<div class="{quality_class}">{quality_msg}</div>', unsafe_allow_html=True)
                    
                    # Strategic Recommendations based on cluster distribution
                    siap_riset_pct = interpretation_df[interpretation_df['Interpretasi'] == 'Siap Riset']['Jumlah Mahasiswa'].sum() / len(labels) * 100 if len(interpretation_df[interpretation_df['Interpretasi'] == 'Siap Riset']) > 0 else 0
                    perlu_pendampingan_pct = interpretation_df[interpretation_df['Interpretasi'] == 'Perlu Pendampingan']['Jumlah Mahasiswa'].sum() / len(labels) * 100 if len(interpretation_df[interpretation_df['Interpretasi'] == 'Perlu Pendampingan']) > 0 else 0
                    
                    st.markdown("### 🎓 Rekomendasi Strategis untuk Kampus")
                    
                    recommendations = []
                    
                    if siap_riset_pct < 30:
                        recommendations.append("📚Tingkatkan Pelatihan Hanya {:.1f}% mahasiswa yang siap riset. Perlu pelatihan intensif untuk meningkatkan kesiapan riset secara keseluruhan.".format(siap_riset_pct))
                    
                    if perlu_pendampingan_pct > 40:
                        recommendations.append("🏫 Workshop Metodologi Dasar: {:.1f}% mahasiswa membutuhkan pendampingan. Disarankan mengadakan workshop metodologi penelitian dasar secara rutin.".format(perlu_pendampingan_pct))
                    
                    if metrics['Cluster Balance Ratio'] < 0.3:
                        recommendations.append("⚖️ Distribusi Tidak Seimbang: Ukuran cluster sangat bervariasi. Pertimbangkan program khusus untuk cluster minoritas.")
                    
                    if metrics['Entropy Index (Gender)'] > 1.5:
                        recommendations.append("🔄 Keragaman Gender Tinggi: Distribusi gender dalam cluster beragam, pastikan program pelatihan inklusif untuk semua.")
                    
                    if not recommendations:
                        recommendations.append("✅ Kondisi Baik: Distribusi cluster dan kesiapan riset mahasiswa dalam kondisi baik. Lanjutkan program yang ada.")
                    
                    for rec in recommendations:
                        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                    
                    # Data with Labels
                    with st.expander("📋 Lihat Data Lengkap dengan Label Cluster"):
                        df_result = st.session_state['original_data'].copy()
                        df_result['cluster'] = labels
                        
                        # Map interpretations
                        interp_map = {}
                        for idx, row in interpretation_df.iterrows():
                            cluster_num = row['Cluster'].split()[-1]
                            if cluster_num.isdigit():
                                interp_map[int(cluster_num)] = row['Interpretasi']
                        
                        df_result['interpretasi'] = df_result['cluster'].map(interp_map)
                        df_result['interpretasi'] = df_result['interpretasi'].fillna('Noise' if -1 in labels else 'Unknown')
                        
                        st.dataframe(df_result)
                        st.session_state['final_result'] = df_result
                    
                    # Download Section
                    st.subheader("📥 Unduh Hasil Akhir")
                    
                    output_data = st.session_state['final_result'].copy()
                    
                    # Add FE columns
                    df_fe = st.session_state['processed_data_full']
                    fe_cols = ['mean_skor', 'std_skor', 'prop_tinggi', 'prop_rendah', 'skor_stabilitas',
                              'gap_tinggi_rendah', 'pengetahuan_riset', 'keterampilan_teknis', 
                              'kesiapan_kepercayaan', 'kebutuhan_pelatihan', 'kebiasaan_akademik', 
                              'profil_aktif', 's_ready']
                    available_fe_cols = [col for col in fe_cols if col in df_fe.columns]
                    
                    for col in available_fe_cols:
                        output_data[col] = df_fe[col].values
                    
                    # CSV Download
                    csv_buffer = io.StringIO()
                    output_data.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download Hasil Clustering (CSV)",
                            data=csv_data,
                            file_name=f"hasil_clustering_{algorithm.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                    
                    # Summary Report
                    summary_buffer = io.StringIO()
                    summary_buffer.write("="*70 + "\n")
                    summary_buffer.write("LAPORAN HASIL CLUSTERING KESIAPAN RISET MAHASISWA\n")
                    summary_buffer.write("="*70 + "\n\n")
                    summary_buffer.write(f"Algoritma: {algorithm}\n")
                    summary_buffer.write(f"Parameter: {params}\n")
                    summary_buffer.write(f"Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    summary_buffer.write("-"*70 + "\n")
                    summary_buffer.write("METRIK EVALUASI:\n")
                    summary_buffer.write("-"*70 + "\n")
                    for metric, score in metrics.items():
                        summary_buffer.write(f"  {metric:.<50} {score:.4f}\n")
                    summary_buffer.write("\n" + "-"*70 + "\n")
                    summary_buffer.write("INTERPRETASI CLUSTER:\n")
                    summary_buffer.write("-"*70 + "\n\n")
                    summary_buffer.write(interpretation_df.to_string(index=False))
                    summary_buffer.write("\n\n" + "-"*70 + "\n")
                    summary_buffer.write("RINGKASAN ASPEK PER CLUSTER:\n")
                    summary_buffer.write("-"*70 + "\n\n")
                    summary_buffer.write(aspect_summary.to_string(index=False))
                    summary_buffer.write("\n\n" + "="*70 + "\n")
                    summary_buffer.write("REKOMENDASI STRATEGIS:\n")
                    summary_buffer.write("="*70 + "\n\n")
                    for i, rec in enumerate(recommendations, 1):
                        clean_rec = rec.replace('<div class="insight-box">', '').replace('</div>', '').strip()
                        summary_buffer.write(f"{i}. {clean_rec}\n\n")
                    
                    with col2:
                        st.download_button(
                            label="📄 Download Laporan Lengkap (TXT)",
                            data=summary_buffer.getvalue(),
                            file_name=f"laporan_clustering_{algorithm.lower().replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

    else:
        st.info("👈 Silakan unggah file CSV untuk memulai analisis.")
        
        st.markdown("""
        ### 🚀 Fitur Unggulan Aplikasi
        
        **1. 🔍 Preprocessing Detail & Evaluatif**
        - Visualisasi distribusi sebelum & sesudah winsorizing
        - Statistik komprehensif per kolom
        - Alert otomatis untuk outlier >5%
        - Heatmap korelasi fitur
        
        **2. 🎯 Feature Engineering**
        - **Skor Stabilitas**: Mengukur konsistensi jawaban
        - **Gap Tinggi-Rendah**: Keseimbangan keyakinan
        - **Profil Aktif**: Indikator kebiasaan akademik
        - **S-Ready**: Skor komposit kesiapan riset
        
        **3. 🤖 Auto-Recommendation Cerdas**
        - K optimal untuk K-Means (Elbow + Silhouette)
        - Eps & Min-Samples untuk DBSCAN
        - Jumlah cluster optimal untuk Hierarchical
        
        **4. 🧠 Interpretasi Otomatis**
        - Klasifikasi: Siap Riset, Antusias Belum Stabil, Perlu Pendampingan, Ragu & Butuh Bimbingan
        - Rekomendasi spesifik per cluster
        - Insight strategis untuk kampus
        
        **5. 📊 Evaluasi Multi-Metrik**
        - 7 metrik evaluasi (Silhouette, Dunn, Balance Ratio, Entropy, dll)
        - Visualisasi interaktif (Elbow, Dendrogram, 2D PCA)
        - Ringkasan aspek tematik per cluster
        
        **6. 💾 Export Komprehensif**
        - CSV lengkap dengan semua fitur + interpretasi
        - Laporan TXT dengan analisis mendalam
        - Rekomendasi strategis untuk institusi
        
        ---
        
        **📚 Tentang Data:**
        Dataset berisi survei kesiapan riset mahasiswa FMIPA Unpad dengan 15 pertanyaan skala Likert (1-5) yang mencakup:
        - Pengetahuan riset (Q1-Q4)
        - Keterampilan teknis (Q5-Q8)
        - Kesiapan & kepercayaan diri (Q9-Q10)
        - Kebutuhan pelatihan (Q11-Q13)
        - Kebiasaan akademik (Q14-Q15)
        
        **🎓 Tujuan:**
        Membantu institusi memahami profil kesiapan riset mahasiswa untuk merancang program pelatihan yang tepat sasaran.
        """)

if __name__ == "__main__":
    main()  
        