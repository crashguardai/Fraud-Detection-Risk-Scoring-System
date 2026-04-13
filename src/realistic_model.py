"""
Realistic Fraud Detection Model - No Data Leakage

This creates a proper model without data leakage, showing realistic ML performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RealisticFraudDetector:
    """
    Realistic fraud detection model without data leakage
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_clean_data(self, filepath='data/enhanced_fraud_data_model_ready.csv'):
        """Load data and remove leaked features"""
        print("Loading clean data without leakage...")
        
        df = pd.read_csv(filepath)
        
        # Remove features that cause data leakage
        leaked_features = [
            'customer_fraud_count',      # Future information
            'customer_fraud_rate',       # Calculated from fraud count
            'risk_score',                # Customer risk score based on fraud history
            'customer_total_amount',     # Aggregate info
            'segment'                    # Risk-based segment
        ]
        
        print(f"Removing leaked features: {leaked_features}")
        df_clean = df.drop(columns=[col for col in leaked_features if col in df.columns])
        
        # Handle categorical columns
        categorical_columns = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df_clean.columns:
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                df_clean = pd.concat([df_clean, dummies], axis=1)
                df_clean.drop(col, axis=1, inplace=True)
        
        # Separate features and target
        X = df_clean.drop('is_fraud', axis=1)
        y = df_clean['is_fraud']
        
        # Handle NaN values
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Clean data loaded: {X.shape}")
        print(f"Fraud rate: {y.mean():.3f}")
        print(f"Number of features: {len(self.feature_columns)}")
        
        return X, y
    
    def train_realistic_model(self):
        """Train realistic model without data leakage"""
        print("="*80)
        print("TRAINING REALISTIC MODEL (NO DATA LEAKAGE)")
        print("="*80)
        
        # Load clean data
        X, y = self.load_clean_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples (fraud rate: {y_train.mean():.3f})")
        print(f"Test set: {X_test.shape[0]} samples (fraud rate: {y_test.mean():.3f})")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"TRAINING: {model_name}")
            print(f"{'='*60}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Cost analysis
            cost_fp = 10
            cost_fn = 1000
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Total Cost: ${total_cost:,}")
            print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'total_cost': total_cost,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'confusion_matrix': cm,
                'model': model
            }
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model_info = results[best_model_name]
        
        print(f"\n{'='*80}")
        print("BEST REALISTIC MODEL")
        print(f"{'='*80}")
        print(f"Model: {best_model_name}")
        print(f"Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"Precision: {best_model_info['precision']:.4f}")
        print(f"Recall: {best_model_info['recall']:.4f}")
        print(f"F1-Score: {best_model_info['f1_score']:.4f}")
        print(f"AUC: {best_model_info['auc']:.4f}")
        print(f"Total Cost: ${best_model_info['total_cost']:,}")
        print(f"CV F1-Score: {best_model_info['cv_f1_mean']:.4f} (+/- {best_model_info['cv_f1_std'] * 2:.4f})")
        
        # Save best model
        self.model = best_model_info['model']
        joblib.dump(self.model, 'models/realistic_best_model.pkl')
        joblib.dump(self.scaler, 'models/realistic_scaler.pkl')
        
        print(f"\nRealistic model saved as: models/realistic_best_model.pkl")
        print(f"Scaler saved as: models/realistic_scaler.pkl")
        
        return best_model_info, results

def main():
    """Main function"""
    detector = RealisticFraudDetector()
    best_model_info, results = detector.train_realistic_model()
    
    print(f"\n{'='*80}")
    print("REALISTIC MODEL TRAINING COMPLETED!")
    print("="*80)
    print("Key improvements:")
    print("1. Removed data leakage features")
    print("2. More realistic performance metrics")
    print("3. Proper cross-validation")
    print("4. Honest performance assessment")
    
    return detector, best_model_info, results

if __name__ == "__main__":
    detector, best_model_info, results = main()
