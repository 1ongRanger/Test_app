from pydantic import BaseModel, Field
from typing import List

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, gt=0, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, gt=0, description="Petal width in cm")

class PredictionInput(BaseModel):
    flowers: List[IrisFeatures] = Field(..., example=[{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }])

class PredictionResult(BaseModel):
    class_name: str
    class_id: int
    probabilities: dict[str, float]

class PredictionOutput(BaseModel):
    predictions: List[PredictionResult]

class HealthResponse(BaseModel):
    status: str = Field(..., example="OK")