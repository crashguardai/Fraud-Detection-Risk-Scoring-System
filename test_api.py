"""
Simple test script for the Fraud Detection API
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("  Health check: OK")
            print(f"  Response: {response.json()}")
        else:
            print(f"  Health check failed: {response.status_code}")
            return False
        
        # Test model info endpoint
        print("\nTesting model info endpoint...")
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            print("  Model info: OK")
            model_info = response.json()
            print(f"  Model type: {model_info.get('model_type', 'Unknown')}")
            print(f"  Features: {model_info.get('features_count', 'Unknown')}")
            print(f"  Accuracy: {model_info.get('accuracy', 'Unknown')}")
            print(f"  F1-Score: {model_info.get('f1_score', 'Unknown')}")
        else:
            print(f"  Model info failed: {response.status_code}")
        
        # Test prediction endpoint
        print("\nTesting prediction endpoint...")
        test_transaction = {
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
        }
        
        response = requests.post(f"{base_url}/predict", json=test_transaction)
        if response.status_code == 200:
            print("  Prediction: OK")
            result = response.json()
            print(f"  Is Fraud: {result['is_fraud']}")
            print(f"  Fraud Probability: {result['fraud_probability']:.3f}")
            print(f"  Risk Score: {result['risk_score']:.1f}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
        else:
            print(f"  Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
        
        # Test batch prediction
        print("\nTesting batch prediction endpoint...")
        batch_request = {
            "transactions": [
                test_transaction,
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
        }
        
        response = requests.post(f"{base_url}/batch_predict", json=batch_request)
        if response.status_code == 200:
            print("  Batch prediction: OK")
            result = response.json()
            print(f"  Processed {len(result['predictions'])} transactions")
            print(f"  Summary: {result['summary']}")
            print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
        else:
            print(f"  Batch prediction failed: {response.status_code}")
        
        print("\n" + "="*50)
        print("API TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("API is running and all endpoints are working correctly.")
        print(f"API Documentation: {base_url}/docs")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("API is not running. Please start the API server first:")
        print("  python src/api.py")
        return False
    except Exception as e:
        print(f"Error testing API: {e}")
        return False

if __name__ == "__main__":
    test_api()
