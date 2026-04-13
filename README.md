# Fraud Detection System

A comprehensive end-to-end Machine Learning project for detecting fraudulent transactions with realistic performance metrics and production-ready deployment.

## Project Overview

This project demonstrates the complete ML lifecycle from data generation to production deployment, with a focus on **honest performance metrics** and **avoiding data leakage** - a critical lesson in real-world ML engineering.

### Key Achievement
- **From "Perfect" to Realistic**: Transformed a fake 100% accuracy model (due to data leakage) into a production-ready system with 93.17% F1-score
- **Data Leakage Identification**: Discovered and removed features that contained future information
- **Production-Ready Model**: Achieved realistic performance that will actually work in production

## Project Structure

```
fraud-detection/
|-- data/
|   |-- fraud_data.csv                     # Original dataset
|   |-- enhanced_fraud_data.csv           # Enhanced dataset (with data leakage)
|   |-- realistic_fraud_data.csv           # Realistic dataset (no data leakage)
|   |-- *_model_ready.csv                  # Model-ready versions
|-- notebooks/
|   |-- 01_eda.ipynb                       # Exploratory data analysis
|   |-- 02_model_training.ipynb            # Model training experiments
|-- src/
|   |-- data_generation.py                  # Original data generator
|   |-- enhanced_data_generation.py         # Enhanced data generator (data leakage)
|   |-- create_realistic_dataset.py         # Realistic data generator (no leakage)
|   |-- preprocessing.py                   # Data preprocessing
|   |-- model_training.py                   # Original model training
|   |-- improved_model.py                   # Improved model with SMOTE
|   |-- advanced_model.py                   # Advanced model with ensembles
|   |-- realistic_model.py                  # Realistic model training
|   |-- train_realistic_model.py            # Train realistic model
|   |-- api.py                              # Original FastAPI
|   |-- simple_api.py                       # Simplified API
|   |-- enhanced_api.py                     # Enhanced API (data leakage)
|   |-- realistic_api.py                    # Realistic API (production-ready)
|   |-- web_ui.py                           # Flask web interface
|   |-- model_evaluation.py                 # Model evaluation metrics
|-- models/
|   |-- best_model.pkl                      # Original trained model
|   |-- enhanced_best_model.pkl             # Enhanced model (data leakage)
|   |-- realistic_best_model.pkl            # Realistic model (production-ready)
|   |-- *_scaler.pkl                        # Feature scalers
|   |-- *_threshold.pkl                     # Optimal thresholds
|   |-- model_evaluation_results.json       # Performance metrics
|-- static/
|   |-- style.css                           # Web UI styling
|-- templates/
|   |-- index.html                          # Web UI template
|-- requirements.txt                        # Python dependencies
|-- README.md                               # This file
|-- interview_guide.md                      # Interview preparation
|-- PROJECT_SUMMARY.md                      # Project overview
|-- FINAL_METRICS_SUCCESS.md                # Success summary
|-- REALISTIC_MODEL_ANALYSIS.md             # Data leakage analysis
|-- FRONTEND_UPDATE_SUMMARY.md              # Frontend changes
|-- METRICS_IMPROVEMENT_SUMMARY.md          # Metrics improvement story
```

## Tech Stack

### Core Technologies
- **Python 3.14+**: Primary programming language
- **FastAPI**: REST API framework for model serving
- **Flask**: Web interface framework
- **Uvicorn**: ASGI server for FastAPI

### Data Science & ML
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **Imbalanced-learn**: Advanced sampling techniques (SMOTE, undersampling)

### ML Algorithms Used
- **RandomForestClassifier**: Primary fraud detection model
- **GradientBoostingClassifier**: Ensemble learning model
- **LogisticRegression**: Baseline classification model
- **BalancedRandomForest**: Imbalanced data handling
- **EasyEnsemble**: Ensemble method for imbalanced data

### Data Processing
- **StandardScaler**: Feature normalization
- **RobustScaler**: Outlier-resistant scaling
- **SelectKBest**: Feature selection
- **train_test_split**: Data splitting
- **StratifiedKFold**: Cross-validation

### API & Deployment
- **Pydantic**: Data validation and serialization
- **Requests**: HTTP client for API testing
- **CORS Middleware**: Cross-origin resource sharing
- **JSON**: Data interchange format

### Frontend & UI
- **HTML5**: Frontend structure
- **CSS3**: Styling and responsive design
- **JavaScript**: Client-side interactions
- **Bootstrap-inspired**: UI components

### Development & Testing
- **Jupyter Notebook**: Data exploration and experimentation
- **Joblib**: Model serialization
- **Logging**: Application monitoring
- **pytest-style**: Testing approach (implied)

### System Architecture
- **RESTful APIs**: Microservices architecture
- **Multi-port deployment**: Multiple API versions
- **Real-time processing**: Sub-second predictions
- **Health checks**: Service monitoring

### Dependencies
All required packages are listed in `requirements.txt`:
```bash
pandas==2.1.4              # Data manipulation
numpy==1.24.3              # Numerical computing
scikit-learn==1.3.2        # Machine learning
fastapi==0.104.1            # API framework
uvicorn==0.24.0             # ASGI server
flask==2.3.3                # Web framework
imbalanced-learn==0.11.0    # Class imbalance handling
matplotlib==3.7.2           # Visualization
seaborn==0.12.2             # Statistical visualization
plotly==5.17.0              # Interactive plots
pydantic==2.5.0             # Data validation
requests==2.31.0            # HTTP client
joblib==1.3.2               # Model serialization
jupyter==1.0.0              # Notebooks
```

## Features

### Data Generation & Processing
- **Multiple Datasets**: Original, Enhanced (with leakage), and Realistic (production-ready)
- **Feature Engineering**: Comprehensive feature creation with data leakage awareness
- **Realistic Patterns**: Fraud patterns that mimic real-world scenarios

### Model Training & Evaluation
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Class Imbalance Handling**: SMOTE, undersampling, cost-sensitive learning
- **Threshold Optimization**: Business-focused threshold tuning
- **Cross-Validation**: Robust performance estimation

### Production Deployment
- **Multiple APIs**: Original, Enhanced, and Realistic versions
- **Web Interface**: Interactive Flask web UI with real-time predictions
- **Performance Monitoring**: Realistic metrics display
- **Model Comparison**: Side-by-side model performance

### Key Learning Outcomes
- **Data Leakage Detection**: How to identify and avoid data leakage
- **Realistic Performance**: Understanding that 100% metrics are impossible
- **Production Readiness**: What makes a model truly deployable
- **Honest ML**: The importance of realistic expectations

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data (Choose one)

**Original Dataset (2% fraud rate):**
```bash
python src/data_generation.py
```

**Enhanced Dataset (8% fraud rate, contains data leakage):**
```bash
python src/enhanced_data_generation.py
```

**Realistic Dataset (5.5% fraud rate, no data leakage):**
```bash
python src/create_realistic_dataset.py
```

### 3. Train Models

**Original Model:**
```bash
python src/model_training.py
```

**Improved Model:**
```bash
python src/improved_model.py
```

**Realistic Model (Recommended):**
```bash
python src/train_realistic_model.py
```

### 4. Start APIs

**Original API (port 8000):**
```bash
python src/api.py
```

**Realistic API (port 8002) - Production Ready:**
```bash
python src/realistic_api.py
```

### 5. Launch Web Interface
```bash
python src/web_ui.py
```

Then open http://localhost:5000 in your browser.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Data Leakage |
|-------|----------|-----------|--------|----------|-----|--------------|
| Original | 56.15% | 1.82% | 72.50% | 3.55% | 62.54% | No |
| Enhanced (Leakage) | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | **Yes** |
| Realistic | 99.21% | 89.39% | 97.29% | 93.17% | 99.93% | **No** |

## API Endpoints

### Realistic API (http://localhost:8002)
- `GET /` - API information
- `GET /health` - Health check
- `GET /model_info` - Model performance metrics
- `POST /predict` - Make fraud prediction
- `POST /compare_all_models` - Compare all models

### Web UI (http://localhost:5000)
- Interactive form for transaction analysis
- Real-time fraud prediction
- Performance metrics display
- Model comparison features

## Sample API Usage

```python
import requests

# Sample transaction data
transaction = {
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
    "customer_income": 60000.0,
    "customer_mobile_preference": 0.7,
    "customer_home_location_variety": 15.0
}

# Make prediction
response = requests.post('http://localhost:8002/predict', json=transaction)
result = response.json()

print(f"Fraud Probability: {result['fraud_probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Is Fraud: {result['is_fraud']}")
```

## Key Learnings

### 1. Data Leakage is Dangerous
- Easy to accidentally include future information
- Makes models look perfect but fail in production
- Must audit every feature carefully

### 2. Realistic Performance is Valuable
- 93% F1-score is excellent for fraud detection
- Catches 97% of fraud with manageable false positives
- Cost-effective operational model

### 3. Honesty in ML
- Perfect metrics are a red flag
- Realistic expectations enable proper planning
- Production-ready models provide genuine value

## Business Impact

### Realistic Model Performance:
- **Fraud Detection Rate**: 97.29% (catches most fraud)
- **False Positive Rate**: 10.61% (manageable customer friction)
- **Operational Cost**: $31,280 per 20,000 transactions
- **ROI**: Significant fraud prevention vs operational costs

### Production Readiness:
- **Scalable Architecture**: FastAPI + Flask deployment
- **Real-time Processing**: Sub-second prediction times
- **Monitoring**: Performance tracking and health checks
- **User Interface**: Interactive web application

## Interview Preparation

This project covers key ML engineering concepts:

- **Data Quality**: Data leakage detection and prevention
- **Model Selection**: Algorithm comparison and selection
- **Performance Metrics**: Beyond accuracy to business impact
- **Production Deployment**: API development and monitoring
- **Ethical ML**: Honest performance reporting

See `interview_guide.md` for detailed explanations and common interview questions.

## Contributing

This project serves as a comprehensive example of real-world ML engineering. Key areas for extension:

1. **Feature Engineering**: Additional behavioral features
2. **Model Optimization**: Hyperparameter tuning
3. **Deployment**: Containerization and cloud deployment
4. **Monitoring**: Real-time performance tracking
5. **Explainability**: SHAP values for model interpretability

## License

This project is for educational purposes to demonstrate ML engineering best practices.

---

**Note**: The journey from "perfect" to realistic metrics teaches a crucial lesson in ML engineering - honest, realistic performance is more valuable than fake perfection.
