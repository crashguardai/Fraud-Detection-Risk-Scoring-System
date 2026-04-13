"""
Demo Script for Fraud Detection System

This script demonstrates the complete fraud detection pipeline
from data generation to API predictions.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_complete_demo():
    """
    Run a complete demonstration of the fraud detection system
    """
    print("="*80)
    print("FRAUD DETECTION SYSTEM - COMPLETE DEMO")
    print("="*80)
    
    # Step 1: Data Generation
    print("\n1. GENERATING SAMPLE DATA...")
    print("-" * 40)
    
    try:
        from src.data_generation import generate_fraud_dataset, save_dataset
        df = generate_fraud_dataset(n_samples=1000, fraud_ratio=0.02)  # Smaller for demo
        save_dataset(df, 'data/demo_fraud_data.csv')
        print(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraud cases")
    except Exception as e:
        print(f"Data generation failed: {e}")
        return False
    
    # Step 2: Preprocessing
    print("\n2. PREPROCESSING DATA...")
    print("-" * 40)
    
    try:
        from src.preprocessing import FraudDetectionPreprocessor
        preprocessor = FraudDetectionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            'data/demo_fraud_data.csv', test_size=0.3
        )
        print(f"Preprocessed data: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return False
    
    # Step 3: Model Training
    print("\n3. TRAINING MODELS...")
    print("-" * 40)
    
    try:
        from src.model_training import FraudDetectionModelTrainer
        trainer = FraudDetectionModelTrainer()
        
        # Initialize and train models
        models = trainer.initialize_models()
        trainer.train_models(models, X_train, y_train)
        
        # Evaluate models
        trainer.evaluate_all_models(X_test, y_test)
        
        # Get best model
        best_name, best_model, best_score = trainer.get_best_model()
        print(f"Best model: {best_name} (F1-Score: {best_score:.4f})")
        
        # Save models
        trainer.save_models()
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return False
    
    # Step 4: API Demo
    print("\n4. STARTING API SERVER...")
    print("-" * 40)
    
    try:
        import subprocess
        import threading
        
        # Start API server in background
        def start_api():
            subprocess.run([sys.executable, 'src/api.py'], 
                          cwd=os.path.dirname(__file__))
        
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        
        # Wait for API to start
        time.sleep(5)
        
        # Test API health
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print("API server started successfully!")
        else:
            print("API server not responding")
            return False
            
    except Exception as e:
        print(f"API startup failed: {e}")
        return False
    
    # Step 5: Test Predictions
    print("\n5. TESTING PREDICTIONS...")
    print("-" * 40)
    
    try:
        # Test with sample transactions
        test_transactions = [
            {
                "transaction_amount": 1500.00,
                "customer_id": 123,
                "customer_age": 35,
                "customer_tenure_days": 365,
                "merchant_category": "online",
                "transaction_hour": 23,
                "distance_from_home_km": 250.0,
                "distance_from_last_transaction_km": 100.0,
                "devices_used_today": 5,
                "is_mobile_transaction": True,
                "ratio_to_median_purchase_price": 8.5,
                "customer_avg_amount": 75.0,
                "customer_std_amount": 25.0,
                "customer_transaction_count": 50,
                "customer_fraud_count": 0
            },
            {
                "transaction_amount": 25.50,
                "customer_id": 456,
                "customer_age": 42,
                "customer_tenure_days": 730,
                "merchant_category": "retail",
                "transaction_hour": 14,
                "distance_from_home_km": 5.0,
                "distance_from_last_transaction_km": 2.0,
                "devices_used_today": 1,
                "is_mobile_transaction": False,
                "ratio_to_median_purchase_price": 0.8,
                "customer_avg_amount": 45.0,
                "customer_std_amount": 15.0,
                "customer_transaction_count": 100,
                "customer_fraud_count": 0
            }
        ]
        
        for i, transaction in enumerate(test_transactions, 1):
            print(f"\nTesting Transaction {i}:")
            print(f"  Amount: ${transaction['transaction_amount']:.2f}")
            print(f"  Category: {transaction['merchant_category']}")
            print(f"  Hour: {transaction['transaction_hour']}")
            
            # Make prediction
            response = requests.post('http://localhost:8000/predict', json=transaction)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
                print(f"  Fraud Probability: {result['fraud_probability']:.3f}")
                print(f"  Risk Score: {result['risk_score']:.1f}")
                print(f"  Risk Level: {result['risk_level']}")
                print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
            else:
                print(f"  Prediction failed: {response.status_code}")
        
    except Exception as e:
        print(f"Prediction testing failed: {e}")
        return False
    
    # Step 6: Summary
    print("\n6. DEMO SUMMARY")
    print("-" * 40)
    print("Successfully demonstrated:")
    print("  1. Data generation with realistic fraud patterns")
    print("  2. Comprehensive preprocessing and feature engineering")
    print("  3. Model training and evaluation")
    print("  4. API deployment with real-time predictions")
    print("  5. Risk scoring and confidence assessment")
    
    print("\nThe system is ready for production use!")
    print("API Documentation: http://localhost:8000/docs")
    
    return True

def show_sample_predictions():
    """
    Show sample predictions for different transaction types
    """
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS FOR DIFFERENT SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            "name": "High-Risk Online Transaction",
            "data": {
                "transaction_amount": 2500.00,
                "customer_id": 789,
                "customer_age": 28,
                "customer_tenure_days": 30,
                "merchant_category": "online",
                "transaction_hour": 3,
                "distance_from_home_km": 500.0,
                "distance_from_last_transaction_km": 200.0,
                "devices_used_today": 8,
                "is_mobile_transaction": True,
                "ratio_to_median_purchase_price": 15.0,
                "customer_avg_amount": 50.0,
                "customer_std_amount": 20.0,
                "customer_transaction_count": 5,
                "customer_fraud_count": 0
            }
        },
        {
            "name": "Normal Retail Purchase",
            "data": {
                "transaction_amount": 75.99,
                "customer_id": 101,
                "customer_age": 45,
                "customer_tenure_days": 1825,
                "merchant_category": "retail",
                "transaction_hour": 16,
                "distance_from_home_km": 2.0,
                "distance_from_last_transaction_km": 1.0,
                "devices_used_today": 1,
                "is_mobile_transaction": False,
                "ratio_to_median_purchase_price": 1.1,
                "customer_avg_amount": 65.0,
                "customer_std_amount": 30.0,
                "customer_transaction_count": 200,
                "customer_fraud_count": 0
            }
        },
        {
            "name": "Suspicious Travel Transaction",
            "data": {
                "transaction_amount": 850.00,
                "customer_id": 202,
                "customer_age": 33,
                "customer_tenure_days": 400,
                "merchant_category": "travel",
                "transaction_hour": 22,
                "distance_from_home_km": 1500.0,
                "distance_from_last_transaction_km": 800.0,
                "devices_used_today": 3,
                "is_mobile_transaction": True,
                "ratio_to_median_purchase_price": 6.5,
                "customer_avg_amount": 80.0,
                "customer_std_amount": 35.0,
                "customer_transaction_count": 25,
                "customer_fraud_count": 1
            }
        }
    ]
    
    try:
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print("-" * 30)
            
            response = requests.post('http://localhost:8000/predict', json=scenario['data'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
                print(f"  Confidence: {result['confidence']}")
                print(f"  Risk Level: {result['risk_level']}")
                print(f"  Key indicators: Amount=${scenario['data']['transaction_amount']}, "
                      f"Hour={scenario['data']['transaction_hour']}, "
                      f"Distance={scenario['data']['distance_from_home_km']}km")
            else:
                print(f"  Error: {response.status_code}")
                
    except Exception as e:
        print(f"Sample predictions failed: {e}")

def show_performance_metrics():
    """
    Display model performance metrics
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    try:
        response = requests.get('http://localhost:8000/model_info')
        
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model Type: {model_info['model_type']}")
            print(f"Features Count: {model_info['features_count']}")
            print(f"Accuracy: {model_info['accuracy']:.4f}")
            print(f"F1-Score: {model_info['f1_score']:.4f}")
            print(f"AUC Score: {model_info['auc_score']:.4f}")
        else:
            print("Could not retrieve model information")
            
    except Exception as e:
        print(f"Performance metrics failed: {e}")

if __name__ == "__main__":
    print("Starting Fraud Detection System Demo...")
    
    # Check if API is already running
    api_running = False
    try:
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            api_running = True
            print("API server is already running!")
    except:
        pass
    
    if not api_running:
        # Run complete demo
        success = run_complete_demo()
        
        if success:
            # Show additional demonstrations
            show_sample_predictions()
            show_performance_metrics()
    else:
        # API is running, just show predictions
        show_sample_predictions()
        show_performance_metrics()
    
    print("\n" + "="*80)
    print("DEMO COMPLETED!")
    print("="*80)
    print("To explore further:")
    print("  - Visit http://localhost:8000/docs for API documentation")
    print("  - Check the notebooks in the notebooks/ directory")
    print("  - Review the interview guide for detailed explanations")
    print("  - Examine the source code in src/ directory")
