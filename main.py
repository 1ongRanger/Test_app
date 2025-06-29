from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from .models import IrisModel
from .schemas import HealthResponse, PredictionInput, PredictionOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Flower Classification API",
    description="API for classifying iris flowers using machine learning",
    version="1.0.0",
    openapi_tags=[{
        "name": "health",
        "description": "Health check endpoints"
    }, {
        "name": "prediction",
        "description": "Prediction endpoints"
    }]
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IrisModel()

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Iris Classification Service"}

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check service health"""
    logger.info("Health check requested")
    return {"status": "OK"}

@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
async def predict(input_data: PredictionInput):
    """Classify iris flowers"""
    try:
        predictions = []
        for flower in input_data.flowers:
            features = [
                flower.sepal_length,
                flower.sepal_width,
                flower.petal_length,
                flower.petal_width
            ]
            result = model.predict(features)
            predictions.append({
                "class_name": result["class"],
                "class_id": result["class_id"],
                "probabilities": result["probabilities"]
            })
        
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))