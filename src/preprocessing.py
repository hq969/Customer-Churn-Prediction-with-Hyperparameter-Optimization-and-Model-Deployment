import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df.dropna(inplace=True)

    # Encoding categorical columns
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Geography'] = le.fit_transform(df['Geography'])

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
