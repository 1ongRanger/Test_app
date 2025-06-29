import logging
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)

class IrisModel:
    def __init__(self):
        self.model = None
        self.class_names = ['setosa', 'versicolor', 'virginica']
        self.load_model()

    def load_model(self):
        """Load trained model or train new one"""
        model_path = "iris_model.pkl"
        if os.path.exists(model_path):
            logger.info("Loading pretrained model")
            self.model = joblib.load(model_path)
        else:
            logger.info("Training new model")
            self.train_model()
            joblib.dump(self.model, model_path)

    def train_model(self):
        """Train RandomForest classifier on Iris dataset"""
        iris = load_iris()
        X, y = iris.data, iris.target
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        logger.info("Model training completed")

    def predict(self, features: list) -> dict:
        """Make prediction and return human-readable result"""
        try:
            logger.info(f"Making prediction for features: {features}")
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            return {
                "class": self.class_names[prediction],
                "class_id": int(prediction),
                "probabilities": {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")