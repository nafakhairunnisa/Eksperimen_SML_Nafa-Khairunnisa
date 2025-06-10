import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Load data
    df = pd.read_csv('preprocessing/personality_dataset.csv')
    
    preparation_df = df.copy()
    preparation_df = preparation_df.drop_duplicates()
    
    # Imputasi missing value
    for col in preparation_df.select_dtypes(include='number'):
        preparation_df[col].fillna(preparation_df[col].median(), inplace=True)
    for col in preparation_df.select_dtypes(include='object'):
        preparation_df[col].fillna(preparation_df[col].mode()[0], inplace=True)
    
    # Encoding
    le = LabelEncoder()
    for col in preparation_df.select_dtypes(include='object'):
        preparation_df[col] = le.fit_transform(preparation_df[col])
    
    # Splitting
    X = preparation_df.drop('Personality', axis=1)
    y = preparation_df['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simpan hasil
    os.makedirs('preprocessing/personality_preprocessing', exist_ok=True)
    preparation_df.to_csv('preprocessing/personality_preprocessing/personality_dataset_clean.csv', index=False)  # Nama file sesuai checklist
    X_train.to_csv('preprocessing/personality_preprocessing/X_train.csv', index=False)
    X_test.to_csv('preprocessing/personality_preprocessing/X_test.csv', index=False)
    y_train.to_csv('preprocessing/personality_preprocessing/y_train.csv', index=False)
    y_test.to_csv('preprocessing/personality_preprocessing/y_test.csv', index=False)

if __name__ == "__main__":
    preprocess_data()