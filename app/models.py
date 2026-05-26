from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime
from .database import Base

class Verificacao(Base):
    __tablename__ = "verificacoes"

    id = Column(Integer, primary_key=True, index=True)
    texto = Column(String, index=True)
    classificacao = Column(String)
    confianca = Column(Float)
    fonte = Column(String)
    data_verificacao = Column(DateTime, default=datetime.utcnow)
