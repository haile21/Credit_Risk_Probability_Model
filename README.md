# B5W5: Credit Risk Probability Model for Alternative Data

## Project Summary
This project presents an end-to-end implementation of a credit risk scoring model using alternative transactional data from an eCommerce platform. The goal is to support Bati Bank’s buy-now-pay-later service by predicting a customer's likelihood of default, assigning a credit score, and recommending optimal loan terms.

## Business Context
Bati Bank seeks to offer credit products to customers without traditional credit history. This project uses behavioral data (such as transaction frequency, recency, and monetary value) to infer credit risk and support responsible lending decisions. Since direct default labels are not available, a proxy target variable is engineered based on customer disengagement patterns.

## Objectives
- Define a **proxy credit risk label** using RFM-based customer segmentation.
- Select and engineer predictive features.
- Train machine learning models to:
  - Predict risk probability
  - Assign a credit score
  - Recommend loan amount and duration
- Deploy the best model using FastAPI and Docker.
- Set up CI/CD pipelines with GitHub Actions. 

## Technical Stack
- **Python**, **scikit-learn**, **pandas**, **FastAPI**, **Docker**, **MLflow**
- **CI/CD**: GitHub Actions
- **Unit Testing**: pytest
- **Model Tracking**: MLflow Registry
- **Data Processing**: sklearn Pipelines, feature engineering scripts

 ## Learning Outcomes
- Credit scoring using alternative data
- Proxy target engineering (RFM + Clustering)
- Model training, tuning, and evaluation
- MLOps: API deployment, CI/CD, version control
- Communication of technical findings in business terms

  ## Deliverables
- Trained model with performance metrics
- FastAPI endpoint for live scoring
- MLflow-tracked experiments and model registry
 
 
