import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Tangani missing value
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna('missing', inplace=True)

    # Drop kolom yang tidak dibutuhkan
    df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')

    # Encode kolom kategorikal
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    if 'Embarked' in df.columns:
        le = LabelEncoder()
        df['Embarked'] = le.fit_transform(df['Embarked'])
        print("Unique values Embarked (after encoding):", df['Embarked'].unique())

    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Data berhasil disimpan di: {path}")

if __name__ == "__main__":
    # Hardcoded path input dan output
    input_path = "Kriteria1/titanic/train.csv"
    output_path = "Kriteria1/titanic/train_clean.csv"

    df = load_data(input_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_path)
