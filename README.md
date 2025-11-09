# Wine Quality Prediction API

> **📊 ML Zoomcamp Midterm Project** 
>
> For a helpful mapping to what this project does with respect to the Evaluation Criteria, please see [EVALUATION.md](EVALUATION.md)

## Problem Description

Wine quality assessment is traditionally performed by expert tasters, which is subjective, time-consuming, and expensive. This project addresses the need for automated, objective quality prediction based on physicochemical properties that can be measured in a laboratory.

**Business Problem**: Wine producers need a quick, objective way to predict wine quality during production to:
- Optimize fermentation processes
- Reduce reliance on expensive expert tastings
- Ensure consistent quality before bottling
- Make data-driven decisions about wine classification

**Solution**: A machine learning API that predicts wine quality (high/low) based on 11 physicochemical properties including acidity, sugar content, pH, alcohol percentage, and sulfur dioxide levels. The model achieves 84.5% AUC on test data, providing reliable predictions accessible via REST API.

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Validation AUC**: 0.8318
- **Test AUC**: 0.8454
- **Parameters**: n_estimators=50, max_depth=10, min_samples_leaf=20

## Quick Start

### Local Development with Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run API server
python predict.py

# Test the API
python test_api.py
```

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).
Visit http://localhost:8000/redoc for alternative documentation (ReDoc).

### Docker Deployment

**Docker Hub Repository**: https://hub.docker.com/repository/docker/mattkappa/wine-quality-api

```bash
# Pull and run from Docker Hub (recommended)
docker pull mattkappa/wine-quality-api:latest
docker run -p 8000:8000 mattkappa/wine-quality-api:latest

# Or build locally from source
docker build -t wine-quality-api .
docker run -p 8000:8000 wine-quality-api
```

The API will be available at http://localhost:8000

### Cloud Deployment (Render.com)

This application is deployed on Render.com and accessible at:

**Live Demo**: https://wine-quality-api-zsiz.onrender.com/docs

> **Video Demo**: In case Render.com does not load quickly (as may be the case for free-tier hosting), you can watch the [usage demonstration video](usage_video.mp4) showing the API in action.

#### Testing the Live API - Step by Step

1. **Open the Interactive API Documentation**
   - Visit: https://wine-quality-api-zsiz.onrender.com/docs
   - You'll see the Swagger UI with all available endpoints

2. **Test the Health Check**
   - Click on `GET /health` endpoint
   - Click **"Try it out"**
   - Click **"Execute"**
   - You should see: `{"status": "healthy"}`

3. **Make a Wine Quality Prediction**
   - Click on `POST /predict` endpoint
   - Click **"Try it out"**
   - Use this sample wine data (or modify values):
   ```json
   {
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
   ```
     for another example to get a high quality wine use (or modify values):
     ```json
     {
        "fixed_acidity": 7.3,
        "volatile_acidity": 0.65,
        "citric_acid": 0,
        "residual_sugar": 1.2,
        "chlorides": 0.065,
        "free_sulfur_dioxide": 15,
        "total_sulfur_dioxide": 21,
        "density": 0.9946,
        "ph": 3.39,
        "sulphates": 0.87,
        "alcohol": 12.8
    }
    ```
   - Click **"Execute"**
   - Check the response for prediction and probability

4. **Test with curl (Optional)**
   ```bash
   curl -X POST "https://wine-quality-api-zsiz.onrender.com/predict" \
        -H "Content-Type: application/json" \
        -d '{
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
        }'
   ```

**Note**: First request may take 30-60 seconds as Render spins up the free tier service if it's been idle.

## API Endpoints

### POST /predict
Predict quality for a single wine sample.

**Example Request:**
```json
{
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
```

**Response:**
```json
{
  "prediction": "low",
  "probability": 0.72,
  "model": "RandomForest"
}
```

### POST /predict/batch
Predict quality for multiple wine samples.

### GET /health
Health check endpoint.

## Project Structure

```
.
├── train.py              # Model training script
├── predict.py            # FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── midterm-datasets/     # Wine quality dataset
│   └── winequality-red.csv
└── README.md             # This file
```

## Dataset

**Source**: Wine Quality Dataset from UCI Machine Learning Repository
- **Location**: `midterm-datasets/winequality-red.csv`
- **Samples**: 1,359 (after removing 240 duplicates from original 1,599)
- **Features**: 11 physicochemical properties
  - fixed_acidity, volatile_acidity, citric_acid
  - residual_sugar, chlorides
  - free_sulfur_dioxide, total_sulfur_dioxide
  - density, ph, sulphates, alcohol
- **Target**: Binary classification (quality >= 6 = high quality, else low quality)
- **Format**: Semicolon-separated CSV

The dataset is included in this repository.

## Technology Stack

- **ML Framework**: scikit-learn, XGBoost
- **API Framework**: FastAPI
- **Validation**: Pydantic
- **Containerization**: Docker

## License

Educational project for ML Zoomcamp course.
