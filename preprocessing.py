import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

## =====================================================================================
# PREPROCESSING COMPONENTS
# =====================================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes column names."""
    new_cols = {}
    for col in df.columns:
        new_col = col.lower().strip()
        new_col = new_col.replace(' ', '_').replace('(', '').replace(')', '').replace('?', '')
        new_col = new_col.replace('[', '_').replace(']', '_').replace(',', '')
        
        # Specific mappings
        if 'timestamp' in new_col: new_col = 'timestamp'
        elif 'nama' in new_col and 'opsional' in new_col: new_col = 'nama'
        elif new_col == 'usia': new_col = 'usia'
        elif 'jenis_kelamin' in new_col: new_col = 'jenis_kelamin'
        elif new_col == 'semester': new_col = 'semester'
        elif 'program_studi' in new_col: new_col = 'program_studi'
        elif 'pernah_menulis_artikel' in new_col or 'menulis_artikel_ilmiah' in new_col: new_col = 'q1_menulis_artikel'
        elif 'pernah_mengikuti_seminar' in new_col or 'mengikuti_seminar' in new_col: new_col = 'q2_ikut_seminar'
        elif 'mampu_mencari_jurnal' in new_col: new_col = 'q3_cari_jurnal'
        elif 'memahami_cara_menggunakan_google' in new_col or 'google_scholar' in new_col: new_col = 'q4_paham_database'
        elif 'bisa_menggunakan_software' in new_col: new_col = 'q5_bisa_software'
        elif 'pernah_melakukan_analisis' in new_col: new_col = 'q6_analisis_sederhana'
        elif 'bisa_menulis_laporan' in new_col: new_col = 'q7_menulis_laporan'
        elif 'memahami_metodologi' in new_col: new_col = 'q8_paham_metodologi'
        elif 'percaya_diri_untuk_memulai' in new_col: new_col = 'q9_pd_memulai'
        elif 'merasa_mampu_menyelesaikan' in new_col: new_col = 'q10_pd_tepat_waktu'
        elif 'metodologi_penelitian' in new_col and 'membutuhkan' in new_col: new_col = 'q11_butuh_latih_metodologi'
        elif 'penulisan_akademik' in new_col and 'membutuhkan' in new_col: new_col = 'q12_butuh_latih_penulisan'
        elif 'analisis_data' in new_col and 'membutuhkan' in new_col: new_col = 'q13_butuh_latih_analisis'
        elif 'terbiasa_berdiskusi' in new_col: new_col = 'q14_biasa_diskusi'
        elif 'terbiasa_membaca_jurnal' in new_col: new_col = 'q15_biasa_baca_jurnal'
        elif 'apakah_ada_saran' in new_col or 'saran_lain' in new_col: new_col = 'saran'
        
        new_cols[col] = new_col
    
    df = df.rename(columns=new_cols)
    return df

def get_statistics_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Generate comprehensive statistics for numeric columns."""
    stats = []
    for col in numeric_cols:
        stats.append({
            'Column': col,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Missing': df[col].isnull().sum(),
            'Missing %': (df[col].isnull().sum() / len(df)) * 100
        })
    return pd.DataFrame(stats)

def handle_missing_values(df: pd.DataFrame, numeric_strategy: str, categorical_strategy: str) -> Tuple[pd.DataFrame, Dict]:
    """Handles missing values with detailed reporting."""
    df = df.copy()
    report = {"numeric": {}, "categorical": {}, "total_filled": 0}
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            n_missing = df[col].isnull().sum()
            report["total_filled"] += n_missing
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if numeric_strategy == 'mean':
                    fill_value = df[col].mean()
                elif numeric_strategy == 'median':
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                report["numeric"][col] = {
                    "count": n_missing,
                    "method": numeric_strategy,
                    "value": fill_value
                }
            else:
                if categorical_strategy == 'mode':
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                else:
                    fill_value = "Unknown"
                df[col] = df[col].fillna(fill_value)
                report["categorical"][col] = {
                    "count": n_missing,
                    "method": categorical_strategy,
                    "value": fill_value
                }
    
    return df, report

def handle_outliers_winsorize(df: pd.DataFrame, columns: List[str], lower_limit: float, upper_limit: float) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Caps outliers with detailed statistics."""
    df = df.copy()
    report = {}
    before_data = df[columns].copy()
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            low_val = df[col].quantile(lower_limit)
            high_val = df[col].quantile(upper_limit)
            original_series = df[col].copy()
            df[col] = np.clip(df[col], low_val, high_val)
            capped_count = (original_series != df[col]).sum()
            capped_pct = (capped_count / len(df)) * 100
            
            if capped_count > 0:
                report[col] = {
                    "capped_count": capped_count,
                    "capped_pct": capped_pct,
                    "lower_bound": low_val,
                    "upper_bound": high_val,
                    "mean_before": original_series.mean(),
                    "mean_after": df[col].mean(),
                    "std_before": original_series.std(),
                    "std_after": df[col].std()
                }
    
    return df, report, before_data

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Encodes categorical features with detailed reporting."""
    df_encoded = df.copy()
    report = {"ordinal": [], "one-hot": [], "dropped": []}

    # Ordinal Encoding for 'semester'
    semester_mapping = {
        'Semester 1': 1, 'Semester 2': 2, 'Semester 3': 3, 'Semester 4': 4,
        'Semester 5': 5, 'Semester 6': 6, 'Semester 7': 7, 'Semester 8': 8,
        '> Semester 8': 9
    }
    if 'semester' in df_encoded.columns:
        df_encoded['semester'] = df_encoded['semester'].map(semester_mapping).fillna(0)
        report["ordinal"].append("semester")

    # One-Hot Encoding
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col not in ['nama', 'saran', 'timestamp']:
            unique_vals = df_encoded[col].unique()
            for val in unique_vals[:-1]:
                new_col_name = f"{col}_{str(val)}".replace(" ", "_").lower()
                df_encoded[new_col_name] = (df_encoded[col] == val).astype(int)
            df_encoded.drop(col, axis=1, inplace=True)
            report["one-hot"].append(col)
        else:
            report["dropped"].append(col)
            if col in df_encoded.columns:
                df_encoded.drop(col, axis=1, inplace=True)
        
    return df_encoded.select_dtypes(include=np.number), report

def scale_numeric_features(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Scales features with before/after data."""
    df_scaled = df.copy()
    before_data = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=np.number).columns
    report = {"method": method, "columns": list(numeric_cols)}

    for col in numeric_cols:
        X = df_scaled[col].values
        if method == 'standard':
            mean = np.mean(X)
            std = np.std(X)
            df_scaled[col] = (X - mean) / (std if std > 0 else 1)
        elif method == 'minmax':
            min_val = np.min(X)
            max_val = np.max(X)
            range_val = max_val - min_val
            df_scaled[col] = (X - min_val) / (range_val if range_val > 0 else 1)
            
    return df_scaled, report, before_data

# =====================================================================================
# FEATURE ENGINEERING ENHANCED
# =====================================================================================

def feature_engineering_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with additional composite features."""
    df_fe = df.copy()
    
    q_cols = [col for col in df_fe.columns if col.startswith('q') and any(char.isdigit() for char in col)]
    
    if len(q_cols) > 0:
        # Basic features
        df_fe['mean_skor'] = df_fe[q_cols].mean(axis=1)
        df_fe['std_skor'] = df_fe[q_cols].std(axis=1)
        df_fe['prop_tinggi'] = (df_fe[q_cols] >= 4).sum(axis=1) / len(q_cols)
        df_fe['prop_rendah'] = (df_fe[q_cols] <= 2).sum(axis=1) / len(q_cols)
        
        # Stability score
        df_fe['skor_stabilitas'] = 1 - (df_fe['std_skor'] / (df_fe['mean_skor'] + 1e-10))
        df_fe['skor_stabilitas'] = df_fe['skor_stabilitas'].clip(0, 1)
        
        # Gap between high and low
        df_fe['gap_tinggi_rendah'] = df_fe['prop_tinggi'] - df_fe['prop_rendah']
        
        # Thematic scores
        knowledge_cols = [col for col in q_cols if any(f'q{i}_' in col for i in range(1, 5))]
        if knowledge_cols:
            df_fe['pengetahuan_riset'] = df_fe[knowledge_cols].mean(axis=1)
        
        skills_cols = [col for col in q_cols if any(f'q{i}_' in col for i in range(5, 9))]
        if skills_cols:
            df_fe['keterampilan_teknis'] = df_fe[skills_cols].mean(axis=1)
        
        readiness_cols = [col for col in q_cols if any(f'q{i}_' in col for i in range(9, 11))]
        if readiness_cols:
            df_fe['kesiapan_kepercayaan'] = df_fe[readiness_cols].mean(axis=1)
        
        training_cols = [col for col in q_cols if any(f'q{i}_' in col for i in range(11, 14))]
        if training_cols:
            df_fe['kebutuhan_pelatihan'] = df_fe[training_cols].mean(axis=1)
        
        habits_cols = [col for col in q_cols if any(f'q{i}_' in col for i in range(14, 16))]
        if habits_cols:
            df_fe['kebiasaan_akademik'] = df_fe[habits_cols].mean(axis=1)
            # Active profile
            df_fe['profil_aktif'] = df_fe[habits_cols].mean(axis=1)
        
        # Composite readiness score
        if all(col in df_fe.columns for col in ['pengetahuan_riset', 'keterampilan_teknis', 'kesiapan_kepercayaan', 'kebutuhan_pelatihan', 'kebiasaan_akademik']):
            df_fe['s_ready'] = (
                0.35 * df_fe['pengetahuan_riset'] +
                0.25 * df_fe['keterampilan_teknis'] +
                0.20 * df_fe['kesiapan_kepercayaan'] -
                0.10 * df_fe['kebutuhan_pelatihan'] +
                0.10 * df_fe['kebiasaan_akademik']
            )
    
    return df_fe

def perform_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """PCA from scratch."""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    projection_matrix = eigenvectors[:, :n_components]
    X_pca = X_centered.dot(projection_matrix)
    explained_variance_ratio = np.sum(eigenvalues[:n_components]) / np.sum(eigenvalues)
    
    return X_pca, eigenvalues, explained_variance_ratio
