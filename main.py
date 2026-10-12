from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from SRC.config import APP_NAME, VERSION, API_SECRET_KEY
from SRC.models.inference import TextClassifier, TextClassifierGlove
from SRC.models.schemas import TextRequest, PredictionResponse
from fastapi.security import APIKeyHeader

# Intialize Classifiers
Classifier = TextClassifier()   # BOW_SVM_model

Classifier_Glove = TextClassifierGlove()    # # GLOVE_Twitter_SVM_model

# Intialize Fastapi APP
app = FastAPI(title=APP_NAME,
              version=VERSION,
              description='API for Sentiment Analysis for Twitter Data')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name='X-API-Key')

async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="You are not authorized to use this API")
    
    return api_key

@app.get('/', tags=['Healthy'], description='Healthy status check endpoint')
async def home(api_key: str=Depends(verify_api_key)):
    return {
        'App_Name': APP_NAME, 
        'Version': VERSION,
        'Status': 'Up & Running'
    }
    



@app.post("/predict", tags=['Predictions'], description="Predict sentiment of the given text", response_model=PredictionResponse)
async def predict(request: TextRequest ,api_key: str = Depends(verify_api_key)):
    
    try:
        predictions = Classifier.predict(request.texts)
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    


@app.post("/predict/glove", tags=["Predictions"], description="Predict sentiment of the given text using Glove embeddings", response_model=PredictionResponse)
async def predict_glove(request: TextRequest ,api_key: str = Depends(verify_api_key)):
    
    try:
        predictions = Classifier_Glove.predict(request.texts)
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
