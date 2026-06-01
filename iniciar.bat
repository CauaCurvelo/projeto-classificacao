@echo off
title Sistema de Fact-Checking - Iniciador
echo ===================================================
echo     SISTEMA DE CLASSIFICACAO DE NOTICIAS FALSAS
echo ===================================================
echo.

echo [1/4] Verificando e instalando dependencias (se necessario)...
pip install -r requirements.txt -q

IF NOT EXIST "ml\modelo.pkl" (
    echo.
    echo [2/4] ARTEFATO NAO ENCONTRADO: O modelo de IA ainda nao foi treinado.
    echo       Iniciando treinamento automatico (GridSearchCV)...
    echo       AVISO: Isso pode levar de 1 a 2 minutos. Nao feche a janela!
    echo.
    python dataset\build_dataset.py
    python ml\train.py
) ELSE (
    echo [2/4] Modelo de IA (modelo.pkl) ja esta pronto e carregado.
)

echo.
echo [3/4] Iniciando o Servidor de Inteligencia Artificial (FastAPI)...
:: Abre o Uvicorn em uma nova janela de terminal para não travar este script
start "Backend - FastAPI" cmd /k "uvicorn app.main:app --port 8000"

echo.
echo [4/4] Aguardando o servidor ligar e abrindo a interface...
:: Pausa de 3 segundos invisível para garantir que a porta 8000 subiu
timeout /t 3 /nobreak > nul

:: Abre o arquivo HTML direto no navegador padrão do Windows
start static\index.html

echo.
echo ===================================================
echo   SISTEMA PRONTO! Voce pode usar o navegador agora.
echo ===================================================
pause
