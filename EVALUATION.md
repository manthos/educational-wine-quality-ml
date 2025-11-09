# Project Evaluation - ML Zoomcamp Midterm

This document maps the project deliverables to the evaluation criteria.

## Evaluation Criteria Checklist

### 1. Problem Description

**Where to check**: See [README.md - Problem Description](README.md#problem-description)

---

### 2. EDA

**Where to check**: See [midterm.ipynb](midterm.ipynb) - Section "# EDA, feature importance analysis" it includes:
- Correlation analysis with heatmap
- Feature importance analysis using Random Forest
- Distribution plots for all features
- Box plots for outlier detection
- Pairplot analysis
- Target variable analysis
- Missing values check
- Duplicate detection

---

### 3. Model Training

**Where to check**: See [midterm.ipynb](midterm.ipynb) - Section "# Model selection process and parameter tuning"

**Models Trained:**
1. **Decision Tree**
   - Parameters tuned: `max_depth` [6, 10, 15, 20]
   
2. **Random Forest**
   - Parameters tuned: 
     - `n_estimators` [10, 50, 100]
     - `max_depth` [10, 15, 20]
     - `min_samples_leaf` [1, 5, 10, 20]
   
3. **XGBoost**
   - Parameters tuned:
     - `eta` [0.1, 0.3, 0.5]
     - `max_depth` [3, 6, 10]
     - `min_child_weight` [1, 5, 10]

**Best Model**: Random Forest (AUC: 0.8318 validation, 0.8454 test)

---

### 4. Exporting Notebook to Script

**Where to check**: [train.py](train.py)
- Complete training pipeline extracted from notebook
- Includes data loading, cleaning, model training
- Saves model to `wine_quality_model.pkl`
- Saves DictVectorizer to `wine_quality_dv.pkl`

---

### 5. Reproducibility

**Where to check**: https://github.com/manthos/educational-wine-quality-ml
- Dataset committed: [midterm-datasets/winequality-red.csv](midterm-datasets/winequality-red.csv)
- All dependencies listed in [requirements.txt](requirements.txt)
- Clear execution instructions in [README.md](README.md)
- Notebook fully executable (all cells run successfully)
- `train.py` script runs without errors

---

### 6. Model Deployment

**Where to check**: [predict.py](predict.py)
- FastAPI web service implementation
- Endpoints: `/predict`, `/predict/batch`, `/health`, `/`
- Pydantic validation for inputs
- Interactive documentation at `/docs` and `/redoc`

---

### 7. Dependency and Environment Management

**Where to check**: https://github.com/manthos/educational-wine-quality-ml  
- Dependencies file: [requirements.txt](requirements.txt)
- Virtual environment instructions in [README.md - Quick Start](README.md#local-development-with-virtual-environment)
- Clear installation steps:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

---

### 8. Containerization

**Where to check**:
- [Dockerfile](Dockerfile) provided
- [.dockerignore](.dockerignore) for optimized builds
- README includes Docker instructions: [README.md - Docker Deployment](README.md#docker-deployment)
- Published to Docker Hub: https://hub.docker.com/repository/docker/mattkappa/wine-quality-api
- Build and run commands documented:
  ```bash
  docker build -t wine-quality-api .
  docker run -p 8000:8000 wine-quality-api
  ```

---

### 9. Cloud Deployment

**Where to check**:
- Deployed to Render.com free tier (** note that first access may require 20 seconds or more to bring up the service because it is free/unpaid **)
- Live URL: https://wine-quality-api-zsiz.onrender.com/docs
- **Video demonstration**: [usage_video.mp4](usage_video.mp4) shows the API in action (helpful if Render.com is slow to load)
- Step-by-step testing instructions in [README.md - Cloud Deployment](README.md#cloud-deployment-rendercom)
- All deployment files included (Dockerfile works on Render)
- Interactive Swagger UI accessible at live URL

---
