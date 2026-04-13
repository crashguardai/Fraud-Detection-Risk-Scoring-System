"""
FastAPI for Fraud Detection Model Deployment

This module provides a REST API endpoint for real-time fraud detection predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud detection and risk scoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_loaded = False

class TransactionData(BaseModel):
    """
    Pydantic model for transaction data input
    """
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    customer_id: int = Field(..., ge=1, description="Customer identifier")
    customer_age: int = Field(..., ge=18, le=120, description="Customer age")
    customer_tenure_days: int = Field(..., ge=0, description="Customer tenure in days")
    merchant_category: str = Field(..., description="Merchant category")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from home in km")
    distance_from_last_transaction_km: float = Field(..., ge=0, description="Distance from last transaction in km")
    devices_used_today: int = Field(..., ge=1, description="Number of devices used today")
    is_mobile_transaction: bool = Field(..., description="Whether transaction is from mobile device")
    ratio_to_median_purchase_price: float = Field(..., gt=0, description="Ratio to median purchase price")
    customer_avg_amount: float = Field(..., gt=0, description="Customer's average transaction amount")
    customer_std_amount: float = Field(..., ge=0, description="Standard deviation of customer's transactions")
    customer_transaction_count: int = Field(..., ge=1, description="Total customer transactions")
    customer_fraud_count: int = Field(..., ge=0, description="Customer's previous fraud count")

class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response
    """
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: str
    timestamp: str
    processing_time_ms: float

class BatchPredictionRequest(BaseModel):
    """
    Pydantic model for batch prediction request
    """
    transactions: List[TransactionData]

class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction response
    """
    predictions: List[PredictionResponse]
    summary: Dict[str, int]
    processing_time_ms: float

class ModelInfo(BaseModel):
    """
    Pydantic model for model information
    """
    model_type: str
    model_version: str
    features_count: int
    training_date: str
    accuracy: float
    f1_score: float
    auc_score: float

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor
    
    Returns:
        tuple: (model, preprocessor)
    """
    global model, preprocessor, model_loaded
    
    try:
        # Load model
        model = joblib.load('models/best_model.pkl')
        logger.info("Model loaded successfully")
        
        # Load preprocessor
        preprocessor = joblib.load('models/preprocessor.pkl')
        logger.info("Preprocessor loaded successfully")
        
        model_loaded = True
        return model, preprocessor
        
    except FileNotFoundError as e:
        logger.error(f"Model or preprocessor file not found: {e}")
        raise HTTPException(status_code=500, detail="Model files not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def preprocess_transaction_data(transaction_data: TransactionData) -> pd.DataFrame:
    """
    Preprocess transaction data for prediction
    
    Args:
        transaction_data: Transaction data from API request
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    # Convert to DataFrame
    data = transaction_data.dict()
    df = pd.DataFrame([data])
    
    # Feature engineering (same as preprocessing pipeline)
    current_time = datetime.now()
    
    # Time-based features
    df['transaction_time'] = current_time
    df['transaction_day_of_week'] = current_time.weekday()
    df['transaction_month'] = current_time.month
    df['is_weekend'] = df['transaction_day_of_week'] >= 5
    df['is_night_time'] = df['transaction_hour'].between(22, 6)
    df['is_month_end'] = 1 if current_time.day == 31 else 0
    df['is_month_start'] = 1 if current_time.day == 1 else 0
    
    # Amount-based features
    df['log_transaction_amount'] = np.log1p(df['transaction_amount'])
    df['amount_category'] = pd.cut(df['transaction_amount'], 
                                  bins=[0, 10, 50, 100, 500, float('inf')],
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Distance-based features
    df['total_distance_km'] = df['distance_from_home_km'] + df['distance_from_last_transaction_km']
    df['distance_ratio'] = df['distance_from_last_transaction_km'] / (df['distance_from_home_km'] + 1e-6)
    
    # Customer behavior features
    df['customer_fraud_rate'] = df['customer_fraud_count'] / (df['customer_transaction_count'] + 1e-6)
    df['customer_experience'] = pd.cut(df['customer_transaction_count'],
                                     bins=[0, 5, 20, 50, float('inf')],
                                     labels=['New', 'Regular', 'Experienced', 'Very Experienced'])
    
    # Risk scoring features
    df['is_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 3).astype(int)
    df['is_very_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 5).astype(int)
    
    # Device usage risk
    df['is_multiple_devices'] = (df['devices_used_today'] > 3).astype(int)
    
    # Time-based risk categories
    df['time_risk_category'] = pd.cut(df['transaction_hour'],
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                    ordered=False)
    
    # Encode categorical features (same as preprocessing pipeline)
    ordinal_mappings = {
        'amount_category': {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'customer_experience': {'New': 0, 'Regular': 1, 'Experienced': 2, 'Very Experienced': 3},
        'time_risk_category': {'Night': 2, 'Morning': 0, 'Afternoon': 1, 'Evening': 1}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-hot encode merchant_category
    if 'merchant_category' in df.columns:
        dummies = pd.get_dummies(df['merchant_category'], prefix='merchant_category', drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop('merchant_category', axis=1, inplace=True)
    
    # Remove unnecessary columns
    columns_to_drop = ['transaction_time']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Ensure all expected features are present
    expected_features = preprocessor.feature_columns
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    
    # Keep only expected features in correct order
    df = df[expected_features]
    
    # Scale features
    df_scaled = preprocessor.scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=expected_features)
    
    return df_scaled

def calculate_risk_score(fraud_probability: float) -> tuple:
    """
    Calculate risk score and risk level from fraud probability
    
    Args:
        fraud_probability: Probability of fraud
        
    Returns:
        tuple: (risk_score, risk_level, confidence)
    """
    # Risk score (0-100)
    risk_score = fraud_probability * 100
    
    # Risk level classification
    if fraud_probability < 0.1:
        risk_level = "Low"
    elif fraud_probability < 0.3:
        risk_level = "Medium"
    elif fraud_probability < 0.7:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    # Confidence level
    if fraud_probability < 0.1 or fraud_probability > 0.9:
        confidence = "High"
    elif fraud_probability < 0.2 or fraud_probability > 0.8:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return risk_score, risk_level, confidence

@app.on_event("startup")
async def startup_event():
    """
    Load model and preprocessor on startup
    """
    logger.info("Starting up Fraud Detection API...")
    load_model_and_preprocessor()
    logger.info("API startup completed successfully!")

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info")
async def get_model_info():
    """
    Get model information
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load evaluation results
        with open('models/model_evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
        
        # Get best model results
        best_model_name = max(eval_results.keys(), 
                            key=lambda x: eval_results[x]['f1_score'])
        best_results = eval_results[best_model_name]
        
        model_info = ModelInfo(
            model_type=type(model).__name__,
            model_version="1.0.0",
            features_count=len(preprocessor.feature_columns),
            training_date="2024-01-01",  # This should come from model metadata
            accuracy=best_results['accuracy'],
            f1_score=best_results['f1_score'],
            auc_score=best_results['auc']
        )
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction_data: TransactionData):
    """
    Predict fraud for a single transaction
    
    Args:
        transaction_data: Transaction data
        
    Returns:
        PredictionResponse: Fraud prediction with risk assessment
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Preprocess data
        processed_data = preprocess_transaction_data(transaction_data)
        
        # Make prediction
        fraud_probability = model.predict_proba(processed_data)[0, 1]
        is_fraud = fraud_probability > 0.5  # Default threshold
        
        # Calculate risk metrics
        risk_score, risk_level, confidence = calculate_risk_score(fraud_probability)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_probability),
            risk_score=float(risk_score),
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=float(processing_time)
        )
        
        # Log prediction (for monitoring)
        logger.info(f"Prediction made: fraud_prob={fraud_probability:.3f}, "
                   f"is_fraud={is_fraud}, processing_time={processing_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_fraud_batch(batch_request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions
    
    Args:
        batch_request: Batch of transaction data
        
    Returns:
        BatchPredictionResponse: Batch predictions with summary
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    predictions = []
    
    try:
        for transaction_data in batch_request.transactions:
            # Preprocess data
            processed_data = preprocess_transaction_data(transaction_data)
            
            # Make prediction
            fraud_probability = model.predict_proba(processed_data)[0, 1]
            is_fraud = fraud_probability > 0.5
            
            # Calculate risk metrics
            risk_score, risk_level, confidence = calculate_risk_score(fraud_probability)
            
            # Create response
            prediction = PredictionResponse(
                is_fraud=bool(is_fraud),
                fraud_probability=float(fraud_probability),
                risk_score=float(risk_score),
                risk_level=risk_level,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=0.0  # Not tracking individual processing time for batch
            )
            
            predictions.append(prediction)
        
        # Calculate summary
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        total_count = len(predictions)
        
        summary = {
            "total_transactions": total_count,
            "fraud_transactions": fraud_count,
            "legitimate_transactions": total_count - fraud_count,
            "fraud_rate": fraud_count / total_count if total_count > 0 else 0,
            "average_risk_score": sum(p.risk_score for p in predictions) / total_count if total_count > 0 else 0
        }
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time_ms=float(processing_time)
        )
        
        logger.info(f"Batch prediction completed: {total_count} transactions, "
                   f"{fraud_count} fraud detected, processing_time={processing_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

@app.post("/webhook/transaction")
async def transaction_webhook(transaction_data: TransactionData, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for real-time transaction processing
    
    This endpoint can be called by payment systems to check transactions in real-time.
    It processes the prediction asynchronously and can trigger additional actions.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        prediction = await predict_fraud(transaction_data)
        
        # Add background task for logging/alerting
        background_tasks.add_task(log_transaction, transaction_data, prediction)
        
        # If high risk, could trigger additional alerts
        if prediction.risk_level in ["High", "Very High"]:
            background_tasks.add_task(send_alert, transaction_data, prediction)
        
        return {
            "status": "processed",
            "prediction": prediction.dict(),
            "message": "Transaction processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

async def log_transaction(transaction_data: TransactionData, prediction: PredictionResponse):
    """
    Log transaction for monitoring and analysis
    
    Args:
        transaction_data: Transaction data
        prediction: Prediction result
    """
    # This would typically log to a database or monitoring system
    logger.info(f"Transaction logged: amount={transaction_data.transaction_amount}, "
               f"risk_level={prediction.risk_level}, fraud_prob={prediction.fraud_probability:.3f}")

async def send_alert(transaction_data: TransactionData, prediction: PredictionResponse):
    """
    Send alert for high-risk transactions
    
    Args:
        transaction_data: Transaction data
        prediction: Prediction result
    """
    # This would typically send an email, SMS, or push notification
    logger.warning(f"HIGH RISK ALERT: amount=${transaction_data.transaction_amount}, "
                  f"risk_score={prediction.risk_score:.1f}, "
                  f"customer_id={transaction_data.customer_id}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
