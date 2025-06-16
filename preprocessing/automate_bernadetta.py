import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Tangani missing value - sesuaikan dengan notebook
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Drop kolom yang tidak dibutuhkan - sesuai notebook
    df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')
    
    # Encode kolom kategorikal - sesuai notebook
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Data berhasil disimpan di: {path}")

if __name__ == "__main__":
    input_path = "Kriteria1/train.csv"
    output_path = "Kriteria1/titanic/train_clean.csv"
    
    df = load_data(input_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_path)