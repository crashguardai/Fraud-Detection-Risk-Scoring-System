"""
Web UI Server for Fraud Detection System

This provides a simple web interface for testing the fraud detection API.
"""

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Configure Flask
app.secret_key = 'fraud_detection_secret_key'
app.template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
app.static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

@app.route('/')
def index():
    """Main page with the fraud detection form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web UI"""
    try:
        # Forward request to the realistic API server
        api_url = 'http://localhost:8002/predict'
        
        # Convert Flask request to JSON
        transaction_data = {
            'transaction_amount': float(request.json.get('transaction_amount', 0)),
            'customer_id': int(request.json.get('customer_id', 1)),
            'customer_age': int(request.json.get('customer_age', 18)),
            'customer_tenure_days': int(request.json.get('customer_tenure_days', 0)),
            'merchant_category': request.json.get('merchant_category', 'retail'),
            'transaction_hour': int(request.json.get('transaction_hour', 0)),
            'distance_from_home_km': float(request.json.get('distance_from_home_km', 0)),
            'distance_from_last_transaction_km': float(request.json.get('distance_from_last_transaction_km', 0)),
            'devices_used_today': int(request.json.get('devices_used_today', 1)),
            'is_mobile_transaction': bool(request.json.get('is_mobile_transaction', False)),
            'ratio_to_median_purchase_price': float(request.json.get('ratio_to_median_purchase_price', 1.0)),
            'customer_avg_amount': float(request.json.get('customer_avg_amount', 75.0)),
            'customer_income': float(request.json.get('customer_income', 60000.0)),
            'customer_mobile_preference': float(request.json.get('customer_mobile_preference', 0.5)),
            'customer_home_location_variety': float(request.json.get('customer_home_location_variety', 10.0))
        }
        
        # Make request to API
        response = requests.post(api_url, json=transaction_data, timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'API error: {response.status_code}'}), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Cannot connect to Realistic API server. Please ensure the Realistic API is running on http://localhost:8002'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api_health')
def api_health():
    """Check if the API server is running"""
    try:
        response = requests.get('http://localhost:8002/health', timeout=5)
        if response.status_code == 200:
            return jsonify({'status': 'API server is running', 'details': response.json()})
        else:
            return jsonify({'status': 'API server responded with error', 'code': response.status_code})
    except requests.exceptions.ConnectionError:
        return jsonify({'status': 'API server is not running'}), 500
    except Exception as e:
        return jsonify({'status': 'Error checking API', 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Fraud Detection Web UI...")
    print("Web UI will be available at: http://localhost:5000")
    print("Make sure the API server is running at: http://localhost:8000")
    print("\nTo start both servers:")
    print("1. Terminal 1: python src/simple_api.py")
    print("2. Terminal 2: python src/web_ui.py")
    print("3. Open browser: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
