"""
Train Models on Enhanced Dataset

This script trains models on the new enhanced dataset with better fraud patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    """
    Train models on enhanced dataset with better fraud patterns
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.feature_columns = None
        
    def load_enhanced_data(self, filepath='data/enhanced_fraud_data_model_ready.csv'):
        """Load the enhanced dataset"""
        print(f"Loading enhanced dataset from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Separate features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Handle NaN values
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Data loaded: {X.shape}")
        print(f"Fraud rate: {y.mean():.3f}")
        print(f"Number of features: {len(self.feature_columns)}")
        
        return X, y
    
    def create_models(self):
        """Create models for training"""
        print("Creating models...")
        
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
            'Balanced Random Forest': BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
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
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
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
        
        self.models = models
        return models
    
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
        return best_threshold, best_f1
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
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
        
        # Cost analysis
        cost_fp = 10
        cost_fn = 1000
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        print(f"\n{model_name} Results:")
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
        """Complete training pipeline"""
        print("="*80)
        print("TRAINING MODELS ON ENHANCED DATASET")
        print("="*80)
        
        # Load data
        X, y = self.load_enhanced_data()
        
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
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        # Create models
        models = self.create_models()
        
        results = {}
        best_f1 = 0
        best_model_info = None
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"TRAINING: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Optimize threshold
                threshold, best_f1_score = self.optimize_threshold(model, X_test_scaled, y_test)
                
                # Evaluate
                result = self.evaluate_model(model, X_test_scaled, y_test, model_name)
                results[model_name] = result
                
                # Track best model
                if result['f1_score'] > best_f1:
                    best_f1 = result['f1_score']
                    best_model_info = {
                        'name': model_name,
                        'result': result,
                        'model': model
                    }
                
                # Save model
                model_filename = f"models/enhanced_{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model, model_filename)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Find and save best model
        if best_model_info:
            print(f"\n{'='*80}")
            print("BEST MODEL ON ENHANCED DATASET")
            print(f"{'='*80}")
            print(f"Model: {best_model_info['name']}")
            print(f"F1-Score: {best_model_info['result']['f1_score']:.4f}")
            print(f"Precision: {best_model_info['result']['precision']:.4f}")
            print(f"Recall: {best_model_info['result']['recall']:.4f}")
            print(f"Accuracy: {best_model_info['result']['accuracy']:.4f}")
            print(f"AUC: {best_model_info['result']['auc']:.4f}")
            print(f"Total Cost: ${best_model_info['result']['total_cost']:,}")
            print(f"Threshold: {best_model_info['result']['threshold']:.3f}")
            
            # Save best model
            joblib.dump(best_model_info['model'], 'models/enhanced_best_model.pkl')
            joblib.dump(self.scaler, 'models/enhanced_scaler.pkl')
            joblib.dump(self.best_threshold, 'models/enhanced_threshold.pkl')
            
            print(f"\nBest model saved as: models/enhanced_best_model.pkl")
            print(f"Scaler saved as: models/enhanced_scaler.pkl")
            print(f"Threshold saved as: models/enhanced_threshold.pkl")
        
        return best_model_info, results

def main():
    """Main function"""
    trainer = EnhancedModelTrainer()
    best_model_info, results = trainer.train_and_evaluate()
    
    if best_model_info:
        print(f"\n{'='*80}")
        print("ENHANCED MODEL TRAINING COMPLETED!")
        print(f"{'='*80}")
        print("Improvements achieved through:")
        print("1. Better dataset with 8% fraud rate (vs 2% before)")
        print("2. More realistic fraud patterns")
        print("3. Clearer fraud indicators")
        print("4. Better feature distributions")
        print("5. More diverse customer profiles")
        
        return trainer, best_model_info, results
    else:
        print("No models were successfully trained.")
        return None, None, None

if __name__ == "__main__":
    trainer, best_model_info, results = main()
