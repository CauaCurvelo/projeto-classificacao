import pandas as pd
import os

def build_dataset():
    src_path = os.path.join(os.path.dirname(__file__), "Fake.br-Corpus", "preprocessed", "pre-processed.csv")
    if not os.path.exists(src_path):
        return

    df_raw = pd.read_csv(src_path)
    df_novo = pd.DataFrame()
    df_novo['texto'] = df_raw['preprocessed_news']
    df_novo['label'] = df_raw['label'].map({'fake': 0, 'true': 1})
    df_novo = df_novo.dropna()
    
    out_path = os.path.join(os.path.dirname(__file__), "processed_dataset.csv")
    df_novo.to_csv(out_path, index=False)

if __name__ == "__main__":
    build_dataset()
