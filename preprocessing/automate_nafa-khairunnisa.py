import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df, numerical_cols, categorical_cols, target_col):
    """
    Fungsi untuk melakukan preprocessing dataset secara otomatis.

    Parameters:
    - df: DataFrame asli
    - numerical_cols: List of numerical column names
    - categorical_cols: List of categorical column names
    - target_col: Nama kolom target

    Returns:
    - X_train, X_test, y_train, y_test: DataFrame fitur dan target set untuk pelatihan dan pengujian
    """
    # Salin dataset ke dataframe baru
    preparation_df = df.copy()

    # Cek duplikasi data dan missing values
    print("Jumlah data duplikat:", preparation_df.duplicated().sum())
    print("Jumlah missing values per kolom:\n", preparation_df.isna().sum())

    # Drop duplikat
    preparation_df.drop_duplicates(inplace=True)

    # Imputasi missing value dengan median (numerik) dan modus (kategorikal)
    for col in preparation_df.columns:
        if preparation_df[col].dtype == 'object':
            imputasi = preparation_df[col].mode()[0]
        else:
            imputasi = preparation_df[col].median()
        preparation_df[col].fillna(imputasi, inplace=True)

    # Standarisasi
    scaler = StandardScaler()
    preparation_df[numerical_cols] = scaler.fit_transform(preparation_df[numerical_cols])

    # Deteksi outlier menggunakan IQR
    for col in numerical_cols:
        Q1 = preparation_df[col].quantile(0.25)
        Q3 = preparation_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = preparation_df[(preparation_df[col] < lower) | (preparation_df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers")

    # Encoding dataset dengan label encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        preparation_df[col] = le.fit_transform(preparation_df[col])
        label_encoders[col] = le

    # Pisahkan fitur dan label
    X = preparation_df.drop(target_col, axis=1)
    y = preparation_df[target_col]

    # Splitting data 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preparation_df

if __name__ == "__main__":
    # Contoh untuk memanggil fungsi
    df = pd.read_csv("path_to_dataset.csv")  # Gantilah dengan path dataset asli
    numerical_cols = ['numerical_col1', 'numerical_col2']  # Gantilah dengan nama kolom numerik
    categorical_cols = ['categorical_col1', 'categorical_col2']  # Gantilah dengan nama kolom kategorikal
    target_col = 'Personality'  # Gantilah dengan nama kolom target

    X_train, X_test, y_train, y_test, processed_df = preprocess_data(df, numerical_cols, categorical_cols, target_col)
    
    # Simpan dataset hasil preprocessing
    processed_df.to_csv('processed_dataset.csv', index=False)
    print("Dataset telah disimpan ke 'processed_dataset.csv'")