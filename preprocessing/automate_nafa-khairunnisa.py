import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Mendapatkan path absolut ke folder repo
repo_root = Path(__file__).parent.parent
dataset_path = repo_root / 'personality_dataset.csv'

def preprocess_data():
    # Cek file dengan path absolut
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Load data
    df = pd.read_csv('personality_dataset.csv')
    
    # Make copy and drop duplicates
    preparation_df = df.copy()
    preparation_df = preparation_df.drop_duplicates()
    
    # Identify numerical and categorical columns
    numerical_cols = preparation_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = preparation_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Personality')
    
    # Impute missing values
    for col in numerical_cols:
        preparation_df[col].fillna(preparation_df[col].median(), inplace=True)
    for col in categorical_cols:
        preparation_df[col].fillna(preparation_df[col].mode()[0], inplace=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    preparation_df[numerical_cols] = scaler.fit_transform(preparation_df[numerical_cols])
    
    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        preparation_df[col] = le.fit_transform(preparation_df[col])
    
    # Encode target variable
    preparation_df['Personality'] = le.fit_transform(preparation_df['Personality'])
    
    # Splitting
    X = preparation_df.drop('Personality', axis=1)
    y = preparation_df['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create output directory
    os.makedirs('personality_preprocessing', exist_ok=True)
    
    # Save processed data
    preparation_df.to_csv('personality_preprocessing/personality_dataset_clean.csv', index=False)
    X_train.to_csv('personality_preprocessing/X_train.csv', index=False)
    X_test.to_csv('personality_preprocessing/X_test.csv', index=False)
    y_train.to_csv('personality_preprocessing/y_train.csv', index=False)
    y_test.to_csv('personality_preprocessing/y_test.csv', index=False)
    
    print("Preprocessing completed successfully!")
    print(f"Processed data saved to: personality_preprocessing/")

if __name__ == "__main__":
    preprocess_data()