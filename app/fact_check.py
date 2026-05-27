import httpx
import os

GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")

async def check_fact_api(texto: str) -> dict:
    if not GOOGLE_FACT_CHECK_API_KEY:
        return {"encontrado": False, "classificacao": None, "confianca": 0.0}

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": texto, "key": GOOGLE_FACT_CHECK_API_KEY, "languageCode": "pt-BR"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if "claims" in data and len(data["claims"]) > 0:
                    claim = data["claims"][0]
                    review = claim.get("claimReview", [{}])[0]
                    rating = review.get("textualRating", "").lower()
                    
                    is_fake = any(x in rating for x in ["falso", "fake", "mentira", "enganoso"])
                    return {
                        "encontrado": True, 
                        "classificacao": "Não Confiável" if is_fake else "Confiável", 
                        "confianca": 0.98 if is_fake else 0.95
                    }
    except Exception:
        pass
        
    return {"encontrado": False, "classificacao": None, "confianca": 0.0}
