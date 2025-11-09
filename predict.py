"""
FastAPI service for wine quality prediction

Currently using port 8000 for serving, if you run locally please ensure no conflicts or change accordingly.
"""
import pickle
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np

# Load the model and vectorizer
print("Loading model and vectorizer...")
with open('wine_quality_model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('wine_quality_dv.pkl', 'rb') as f_dv:
    dv = pickle.load(f_dv)

print("Model loaded successfully!")

# Create FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality (high/low) based on physicochemical properties",
    version="1.0.0"
)

# Input validation with Pydantic
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(
        ..., 
        ge=0, 
        le=20,
        description="Fixed acidity level (g/dm³)",
        example=7.4
    )
    volatile_acidity: float = Field(
        ..., 
        ge=0, 
        le=2,
        description="Volatile acidity level (g/dm³)",
        example=0.7
    )
    citric_acid: float = Field(
        ..., 
        ge=0, 
        le=1.5,
        description="Citric acid level (g/dm³)",
        example=0.0
    )
    residual_sugar: float = Field(
        ..., 
        ge=0, 
        le=20,
        description="Residual sugar level (g/dm³)",
        example=1.9
    )
    chlorides: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Chlorides level (g/dm³)",
        example=0.076
    )
    free_sulfur_dioxide: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Free sulfur dioxide (mg/dm³)",
        example=11.0
    )
    total_sulfur_dioxide: float = Field(
        ..., 
        ge=0, 
        le=300,
        description="Total sulfur dioxide (mg/dm³)",
        example=34.0
    )
    density: float = Field(
        ..., 
        ge=0.98, 
        le=1.01,
        description="Density (g/cm³)",
        example=0.9978
    )
    ph: float = Field(
        ..., 
        ge=2.5, 
        le=4.5,
        description="pH value",
        example=3.51
    )
    sulphates: float = Field(
        ..., 
        ge=0, 
        le=2.5,
        description="Sulphates level (g/dm³)",
        example=0.56
    )
    alcohol: float = Field(
        ..., 
        ge=8, 
        le=15,
        description="Alcohol percentage (%)",
        example=9.4
    )

    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }


class PredictionResponse(BaseModel):
    quality_prediction: str = Field(
        ...,
        description="Predicted quality category: 'high' (≥6) or 'low' (<6)"
    )
    probability_high_quality: float = Field(
        ...,
        description="Probability of being high quality wine (0-1)"
    )
    probability_low_quality: float = Field(
        ...,
        description="Probability of being low quality wine (0-1)"
    )


@app.get("/")
def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": dv is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    """
    Predict wine quality based on physicochemical properties
    
    - **Returns**: Quality prediction (high/low) and probabilities
    - **Quality threshold**: Rating ≥ 6 is considered "high quality"
    """
    try:
        # Convert input to dictionary
        wine_dict = wine.dict()
        
        # Transform using DictVectorizer
        X = dv.transform([wine_dict])
        
        # Make prediction
        probabilities = model.predict_proba(X)[0]
        prob_low = float(probabilities[0])
        prob_high = float(probabilities[1])
        
        # Determine prediction
        prediction = "high" if prob_high >= 0.5 else "low"
        
        return PredictionResponse(
            quality_prediction=prediction,
            probability_high_quality=prob_high,
            probability_low_quality=prob_low
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(wines: list[WineFeatures]):
    """
    Predict wine quality for multiple wines at once
    
    - **Returns**: List of predictions for each wine
    """
    try:
        results = []
        
        for wine in wines:
            # Convert input to dictionary
            wine_dict = wine.dict()
            
            # Transform using DictVectorizer
            X = dv.transform([wine_dict])
            
            # Make prediction
            probabilities = model.predict_proba(X)[0]
            prob_low = float(probabilities[0])
            prob_high = float(probabilities[1])
            
            # Determine prediction
            prediction = "high" if prob_high >= 0.5 else "low"
            
            results.append({
                "quality_prediction": prediction,
                "probability_high_quality": prob_high,
                "probability_low_quality": prob_low
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Wine Quality Prediction API")
    print("="*60)
    print("\nAPI Documentation available at:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc:      http://localhost:8000/redoc")
    print("\nEndpoints:")
    print("  - POST /predict       - Single prediction")
    print("  - POST /predict/batch - Batch predictions")
    print("  - GET  /health        - Health check")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
