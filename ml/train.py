import pandas as pd
import os
import pickle
import re
import nltk
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

def pre_processar_texto(texto):
    texto = unidecode(str(texto).lower())
    texto = re.sub(r'[^a-z0-9\s\!\?]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def treinar_modelo():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'processed_dataset.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['texto_clean'] = df['texto'].apply(pre_processar_texto)

    X = df['texto_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    stop_words_pt = [unidecode(sw) for sw in stopwords.words('portuguese')]

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(analyzer='word', stop_words=stop_words_pt, max_features=5000)),
            ('char_tfidf', TfidfVectorizer(analyzer='char_wb', max_features=5000))
        ])),
        ('clf', MultinomialNB()) 
    ])

    # GridSearchCV para otimizar os hiperparâmetros matematicamente
    parametros_grid = {
        'features__word_tfidf__ngram_range': [(1, 1), (1, 2)],
        'features__char_tfidf__ngram_range': [(3, 4)],
        'clf__alpha': [0.1, 0.5, 1.0]
    }

    print("Iniciando treinamento com GridSearchCV (isso pode levar 1-2 minutos)...")
    grid_search = GridSearchCV(pipeline, parametros_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    melhor_modelo = grid_search.best_estimator_
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

    y_pred = melhor_modelo.predict(X_test)
    
    print("=== Métricas do Modelo ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

    model_path = os.path.join(os.path.dirname(__file__), 'modelo.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(melhor_modelo, f)

if __name__ == "__main__":
    treinar_modelo()
