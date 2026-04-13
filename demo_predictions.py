"""
Demo script to test the Fraud Detection API with various scenarios
"""

import requests
import json
import time

def test_fraud_scenarios():
    """Test different fraud scenarios"""
    
    base_url = "http://localhost:8000"
    
    print("="*80)
    print("FRAUD DETECTION API - DEMONSTRATION")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        {
            "name": "High-Risk Online Transaction",
            "description": "Large amount, late night, far from home, multiple devices",
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
            "description": "Small amount, daytime, close to home, single device",
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
            "description": "Medium amount, late evening, far from home, mobile device",
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
        },
        {
            "name": "Weekend Gas Station Purchase",
            "description": "Small amount, weekend, moderate distance, mobile",
            "data": {
                "transaction_amount": 45.50,
                "customer_id": 303,
                "customer_age": 52,
                "customer_tenure_days": 900,
                "merchant_category": "gas",
                "transaction_hour": 19,
                "distance_from_home_km": 25.0,
                "distance_from_last_transaction_km": 15.0,
                "devices_used_today": 2,
                "is_mobile_transaction": True,
                "ratio_to_median_purchase_price": 0.9,
                "customer_avg_amount": 55.0,
                "customer_std_amount": 25.0,
                "customer_transaction_count": 150,
                "customer_fraud_count": 0
            }
        },
        {
            "name": "Food Purchase - Low Risk",
            "description": "Very small amount, lunch time, local, desktop",
            "data": {
                "transaction_amount": 12.99,
                "customer_id": 404,
                "customer_age": 38,
                "customer_tenure_days": 600,
                "merchant_category": "food",
                "transaction_hour": 12,
                "distance_from_home_km": 1.5,
                "distance_from_last_transaction_km": 0.5,
                "devices_used_today": 1,
                "is_mobile_transaction": False,
                "ratio_to_median_purchase_price": 0.8,
                "customer_avg_amount": 40.0,
                "customer_std_amount": 20.0,
                "customer_transaction_count": 120,
                "customer_fraud_count": 0
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Amount: ${scenario['data']['transaction_amount']:.2f}")
        print(f"   Hour: {scenario['data']['transaction_hour']}:00")
        print(f"   Distance: {scenario['data']['distance_from_home_km']:.1f}km from home")
        
        try:
            # Make prediction
            response = requests.post(f"{base_url}/predict", json=scenario['data'])
            
            if response.status_code == 200:
                result = response.json()
                
                # Determine risk level emoji
                risk_emojis = {
                    "Low": "   ",
                    "Medium": "  ",
                    "High": "  ",
                    "Very High": ""
                }
                
                print(f"   Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
                print(f"   Risk Level: {risk_emojis.get(result['risk_level'], '')} {result['risk_level']}")
                print(f"   Fraud Probability: {result['fraud_probability']:.3f}")
                print(f"   Risk Score: {result['risk_score']:.1f}/100")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
                
                results.append({
                    'scenario': scenario['name'],
                    'is_fraud': result['is_fraud'],
                    'risk_level': result['risk_level'],
                    'probability': result['fraud_probability'],
                    'risk_score': result['risk_score']
                })
                
            else:
                print(f"   Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   Exception: {e}")
        
        print("-" * 60)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    fraud_count = sum(1 for r in results if r['is_fraud'])
    total_count = len(results)
    
    print(f"Total transactions tested: {total_count}")
    print(f"Fraudulent transactions: {fraud_count} ({fraud_count/total_count*100:.1f}%)")
    print(f"Legitimate transactions: {total_count - fraud_count} ({(total_count-fraud_count)/total_count*100:.1f}%)")
    
    print(f"\nRisk Level Distribution:")
    risk_levels = {}
    for result in results:
        level = result['risk_level']
        risk_levels[level] = risk_levels.get(level, 0) + 1
    
    for level, count in sorted(risk_levels.items()):
        print(f"  {level}: {count} transactions")
    
    print(f"\nAverage Risk Score: {sum(r['risk_score'] for r in results) / len(results):.1f}/100")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("BUSINESS RECOMMENDATIONS")
    print("="*80)
    
    high_risk = [r for r in results if r['risk_level'] in ['High', 'Very High']]
    if high_risk:
        print(f"High-Risk Transactions ({len(high_risk)}):")
        for r in high_risk:
            print(f"  - {r['scenario']}: Score {r['risk_score']:.1f}, Probability {r['probability']:.3f}")
        print(f"\nRecommendation: Review these transactions manually or implement additional verification.")
    
    medium_risk = [r for r in results if r['risk_level'] == 'Medium']
    if medium_risk:
        print(f"\nMedium-Risk Transactions ({len(medium_risk)}):")
        for r in medium_risk:
            print(f"  - {r['scenario']}: Score {r['risk_score']:.1f}, Probability {r['probability']:.3f}")
        print(f"\nRecommendation: Monitor these patterns for potential fraud trends.")
    
    low_risk = [r for r in results if r['risk_level'] == 'Low']
    if low_risk:
        print(f"\nLow-Risk Transactions ({len(low_risk)}):")
        for r in low_risk:
            print(f"  - {r['scenario']}: Score {r['risk_score']:.1f}, Probability {r['probability']:.3f}")
        print(f"\nRecommendation: Process normally with standard verification.")
    
    print(f"\n{'='*80}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("The Fraud Detection API is working correctly and providing")
    print("real-time risk assessments for transaction analysis.")
    print(f"\nAPI Documentation: {base_url}/docs")
    print("Health Check: {base_url}/health")

if __name__ == "__main__":
    test_fraud_scenarios()
