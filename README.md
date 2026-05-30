# 🛡️ Sistema Híbrido de Fact-Checking com Machine Learning

Um sistema *end-to-end* projetado para detecção e classificação de notícias falsas (Fake News). O projeto implementa uma arquitetura híbrida de checagem, cruzando validação externa baseada em fatos (Google Fact Check Tools) com uma Inteligência Artificial local baseada em estilometria de PNL (Processamento de Linguagem Natural).

## 🚀 Arquitetura e Tecnologias

- **Inteligência Artificial (`ml/`)**: Modelo *Multinomial Naive Bayes* treinado sobre vetores TF-IDF (Word & Char N-Grams). Otimizado via `GridSearchCV` e imune ao viés de tamanho de texto (*Length Bias*).
- **Backend API (`app/`)**: Construído com **FastAPI**, implementa *Graceful Degradation* (timeout seguro), validação estrita com **Pydantic** e gerenciamento de memória via `@asynccontextmanager (Lifespan)`.
- **Banco de Dados**: Histórico e Cache de Curto-Circuito utilizando **SQLite** e o ORM **SQLAlchemy**.
- **Frontend (`static/`)**: Interface HTML/JS Vanilla contendo barra de progresso (Termômetro de Veracidade) e injeção dinâmica de disclaimers éticos.
- **Dataset (`dataset/`)**: Alimentado pelo repositório open-source *Fake.br-Corpus* (7.200 artigos reais do contexto brasileiro).

## 📁 Estrutura de Diretórios

```
├── app/
│   ├── main.py           # Core do FastAPI (Rotas, Cache, Lifespan)
│   ├── database.py       # Configuração do SQLite/SQLAlchemy
│   ├── models.py         # Schema relacional do banco de dados
│   └── fact_check.py     # Integração via httpx com a Google Fact Check API
├── dataset/
│   └── build_dataset.py  # Ingestor automatizado do Fake.br-Corpus
├── ml/
│   └── train.py          # Pipeline de treinamento ML (NLTK Stopwords, GridSearchCV)
├── static/
│   └── index.html        # Interface visual do usuário
└── Relatorio_Tecnico.tex # Tese científica descritiva do projeto
```

## ⚙️ Como Instalar e Executar

### 🚀 O Jeito Mais Rápido (1-Clique Windows)
O projeto conta com um script de automação de ambiente. Basta dar 2 cliques no arquivo **`iniciar.bat`**. 
Ele se encarregará de instalar as dependências, treinar a Inteligência Artificial localmente (se for a primeira vez), iniciar o servidor FastAPI e abrir a interface gráfica diretamente no seu navegador padrão.

---

### Execução Manual (Terminal)

Caso prefira rodar passo a passo via terminal:

#### 1. Instalar Dependências
Certifique-se de estar usando o Python 3.10+.
```bash
pip install fastapi uvicorn sqlalchemy pydantic httpx scikit-learn pandas unidecode nltk
```

#### 2. Construir o Dataset e Treinar a IA
Execute os scripts na raiz do projeto para montar a base de dados em CSV e realizar o *Cross-Validation* do modelo.
```bash
python dataset/build_dataset.py
python ml/train.py
```

#### 3. Iniciar o Servidor Backend e Acessar
```bash
uvicorn app.main:app --port 8000
```
Com o servidor rodando, abra o arquivo `static/index.html` em qualquer navegador.

*Trabalho acadêmico rigoroso desenvolvido com foco em desempenho, segurança de dados e alta disponibilidade.*
