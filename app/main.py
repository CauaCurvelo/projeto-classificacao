from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base, get_db
from app.models import Verificacao
from app.fact_check import check_fact_api
from ml.train import pre_processar_texto

Base.metadata.create_all(bind=engine)

# Escopo global para armazenar o modelo
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo de ML na inicialização (Startup)
    caminho_modelo = os.path.join(os.path.dirname(__file__), '..', 'ml', 'modelo.pkl')
    try:
        with open(caminho_modelo, 'rb') as f:
            ml_models["modelo"] = pickle.load(f)
    except Exception:
        ml_models["modelo"] = None
    yield
    # Limpa a memória no encerramento (Shutdown)
    ml_models.clear()

app = FastAPI(title="Fact-Check API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerificacaoRequest(BaseModel):
    texto: str = Field(..., min_length=30, max_length=5000, description="Texto da notícia")

@app.post("/verify")
async def verify_news(request: VerificacaoRequest, db: Session = Depends(get_db)):
    texto = request.texto.strip()
    
    # 1. CACHE (Curto-Circuito Inteligente)
    historico = db.query(Verificacao).filter(Verificacao.texto == texto).first()
    if historico:
        return {
            "texto": historico.texto,
            "classificacao": historico.classificacao,
            "confianca": historico.confianca,
            "fonte": historico.fonte + " (Cache Local)"
        }
    
    # 2. Consulta a API de Fact-Check (Google)
    resultado_api = await check_fact_api(texto)
    
    if resultado_api["encontrado"]:
        classificacao = resultado_api["classificacao"]
        confianca = resultado_api["confianca"]
        fonte = "API Externa"
    else:
        # Fallback para o modelo de ML local
        modelo = ml_models.get("modelo")
        if not modelo:
            return {"erro": "Modelo ML indisponível. Treine o modelo primeiro."}
            
        texto_limpo = pre_processar_texto(texto)
        probabilidades = modelo.predict_proba([texto_limpo])[0]
        prob_verdadeiro = probabilidades[1]
        
        if prob_verdadeiro >= 0.5:
            classificacao = "Confiável"
            confianca = float(prob_verdadeiro)
        else:
            classificacao = "Não Confiável"
            confianca = float(1.0 - prob_verdadeiro)
            
        fonte = "IA Local"

    # Persiste no Banco de Dados SQLite (Dataset da Aplicação)
    nova_verificacao = Verificacao(
        texto=texto,
        classificacao=classificacao,
        confianca=confianca,
        fonte=fonte
    )
    db.add(nova_verificacao)
    db.commit()
    db.refresh(nova_verificacao)

    return {
        "texto": texto,
        "classificacao": classificacao,
        "confianca": confianca,
        "fonte": fonte
    }

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(Verificacao).order_by(Verificacao.id.desc()).limit(10).all()
