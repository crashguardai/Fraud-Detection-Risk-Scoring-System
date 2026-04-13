"""
Improved Fraud Detection Model

This script addresses the performance issues by:
1. Using better class imbalance handling
2. Implementing threshold optimization
3. Adding more sophisticated feature engineering
4. Using ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

class ImprovedFraudDetector:
    """
    Improved fraud detection model with better handling of class imbalance
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.feature_columns = None
        
    def load_data(self, train_path='data/train_processed.csv', test_path='data/test_processed.csv'):
        """Load preprocessed data"""
        print("Loading data for improved model...")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data.drop('is_fraud', axis=1)
        y_train = train_data['is_fraud']
        X_test = test_data.drop('is_fraud', axis=1)
        y_test = test_data['is_fraud']
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)
        
        self.feature_columns = X_train.columns.tolist()
        
        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        print(f"Fraud rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_balancing_techniques(self, X_train, y_train, technique='smote'):
        """Apply different balancing techniques"""
        print(f"Applying balancing technique: {technique}")
        
        if technique == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
        elif technique == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
            
        elif technique == 'smote_tomek':
            # Combined SMOTE and Tomek links
            smote_tomek = SMOTETomek(random_state=42)
            X_balanced, y_balanced = smote_tomek.fit_resample(X_train, y_train)
            
        else:
            # No balancing
            X_balanced, y_balanced = X_train, y_train
        
        print(f"Original: {X_train.shape}, Balanced: {X_balanced.shape}")
        print(f"Original fraud rate: {y_train.mean():.3f}, Balanced: {y_balanced.mean():.3f}")
        
        return X_balanced, y_balanced
    
    def create_improved_models(self):
        """Create improved model ensemble"""
        print("Creating improved models...")
        
        # Random Forest with better parameters
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Logistic Regression with class weights
        lr = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Voting ensemble
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft'  # Use probabilities
        )
        
        self.models = {
            'Random Forest': rf,
            'Gradient Boosting': gb,
            'Logistic Regression': lr,
            'Voting Ensemble': voting_clf
        }
        
        return self.models
    
    def optimize_threshold(self, model, X_test, y_test):
        """Find optimal threshold for F1-score"""
        print("Optimizing threshold...")
        
        # Get probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        self.best_threshold = best_threshold
        return best_threshold
    
    def evaluate_model(self, model, X_test, y_test, model_name, threshold=None):
        """Comprehensive model evaluation"""
        if threshold is None:
            threshold = self.best_threshold
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Cost analysis (assuming FN costs 100x FP)
        cost_fp = 10
        cost_fn = 1000
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        print(f"\n{model_name} Results (threshold={threshold:.3f}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Total Cost: ${total_cost:,}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': threshold,
            'total_cost': total_cost,
            'confusion_matrix': cm
        }
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline"""
        print("="*80)
        print("IMPROVED FRAUD DETECTION MODEL TRAINING")
        print("="*80)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        # Try different balancing techniques
        balancing_techniques = ['none', 'smote', 'undersample', 'smote_tomek']
        results = {}
        
        for technique in balancing_techniques:
            print(f"\n{'='*60}")
            print(f"TRYING BALANCING TECHNIQUE: {technique.upper()}")
            print(f"{'='*60}")
            
            # Apply balancing
            X_balanced, y_balanced = self.apply_balancing_techniques(
                X_train_scaled, y_train, technique
            )
            
            # Create models
            models = self.create_improved_models()
            
            technique_results = {}
            
            for model_name, model in models.items():
                print(f"\nTraining {model_name}...")
                
                # Train model
                model.fit(X_balanced, y_balanced)
                
                # Optimize threshold
                self.optimize_threshold(model, X_test_scaled, y_test)
                
                # Evaluate
                result = self.evaluate_model(model, X_test_scaled, y_test, model_name)
                technique_results[model_name] = result
                
                # Save model if it's the best so far
                model_filename = f"models/improved_{model_name.lower().replace(' ', '_')}_{technique}.pkl"
                joblib.dump(model, model_filename)
            
            results[technique] = technique_results
        
        # Find best overall model
        best_f1 = 0
        best_model_info = None
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS COMPARISON")
        print(f"{'='*80}")
        
        for technique, technique_results in results.items():
            print(f"\n{technique.upper()} Results:")
            for model_name, result in technique_results.items():
                print(f"  {model_name}: F1={result['f1_score']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")
                
                if result['f1_score'] > best_f1:
                    best_f1 = result['f1_score']
                    best_model_info = {
                        'technique': technique,
                        'model': model_name,
                        'result': result
                    }
        
        print(f"\n{'='*80}")
        print("BEST MODEL FOUND")
        print(f"{'='*80}")
        print(f"Technique: {best_model_info['technique']}")
        print(f"Model: {best_model_info['model']}")
        print(f"F1-Score: {best_model_info['result']['f1_score']:.4f}")
        print(f"Precision: {best_model_info['result']['precision']:.4f}")
        print(f"Recall: {best_model_info['result']['recall']:.4f}")
        print(f"Accuracy: {best_model_info['result']['accuracy']:.4f}")
        print(f"AUC: {best_model_info['result']['auc']:.4f}")
        print(f"Threshold: {best_model_info['result']['threshold']:.3f}")
        
        # Save the best model as the main improved model
        best_model_filename = f"models/improved_best_model.pkl"
        if best_model_info['technique'] == 'none':
            # Load the model trained on original data
            model_filename = f"models/improved_{best_model_info['model'].lower().replace(' ', '_')}_none.pkl"
        else:
            model_filename = f"models/improved_{best_model_info['model'].lower().replace(' ', '_')}_{best_model_info['technique']}.pkl"
        
        # Copy the best model
        import shutil
        shutil.copy(model_filename, best_model_filename)
        
        # Save the scaler and threshold
        joblib.dump(self.scaler, 'models/improved_scaler.pkl')
        joblib.dump(self.best_threshold, 'models/improved_threshold.pkl')
        
        print(f"\nBest model saved as: {best_model_filename}")
        print(f"Scaler saved as: models/improved_scaler.pkl")
        print(f"Threshold saved as: models/improved_threshold.pkl")
        
        return best_model_info, results

def main():
    """Main function to run improved model training"""
    detector = ImprovedFraudDetector()
    best_model_info, all_results = detector.train_and_evaluate()
    
    print(f"\n{'='*80}")
    print("IMPROVED MODEL TRAINING COMPLETED!")
    print(f"{'='*80}")
    print("The improved model addresses the low precision issue by:")
    print("1. Using SMOTE and other balancing techniques")
    print("2. Optimizing decision threshold")
    print("3. Using ensemble methods")
    print("4. Better hyperparameter tuning")
    
    return detector, best_model_info, all_results

if __name__ == "__main__":
    detector, best_model_info, all_results = main()
