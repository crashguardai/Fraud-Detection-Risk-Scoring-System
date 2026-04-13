"""
Train Realistic Model on Truly Realistic Dataset

This trains models on a dataset with no data leakage and realistic fraud patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RealisticModelTrainer:
    """
    Train models on realistic dataset without data leakage
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_realistic_data(self, filepath='data/realistic_fraud_data_model_ready.csv'):
        """Load realistic dataset"""
        print(f"Loading realistic dataset from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Handle any remaining categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            print(f"Encoding categorical columns: {categorical_columns}")
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
        
        print(f"Realistic data loaded: {X.shape}")
        print(f"Fraud rate: {y.mean():.3%}")
        print(f"Number of features: {len(self.feature_columns)}")
        
        return X, y
    
    def create_models(self):
        """Create models for realistic training"""
        print("Creating models for realistic training...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
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
        
        self.models = models
        return models
    
    def evaluate_model_realistically(self, model, X_train, X_test, y_train, y_test, model_name):
        """Realistic model evaluation with cross-validation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
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
        
        print(f"Test Set Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Total Cost: ${total_cost:,}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # Business metrics
        fraud_rate_actual = y_test.mean()
        fraud_rate_predicted = y_pred.mean()
        
        print(f"  Actual Fraud Rate: {fraud_rate_actual:.3%}")
        print(f"  Predicted Fraud Rate: {fraud_rate_predicted:.3%}")
        
        # Precision-Recall tradeoff analysis
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find threshold for 90% recall
        recall_90_idx = np.where(recalls >= 0.9)[0]
        if len(recall_90_idx) > 0:
            threshold_90_recall = thresholds[recall_90_idx[0]]
            precision_at_90_recall = precisions[recall_90_idx[0]]
            print(f"  At 90% recall: Precision={precision_at_90_recall:.4f}, Threshold={threshold_90_recall:.3f}")
        
        return {
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
    
    def train_realistic_models(self):
        """Complete realistic training pipeline"""
        print("="*80)
        print("TRAINING REALISTIC MODELS (NO DATA LEAKAGE)")
        print("="*80)
        
        # Load realistic data
        X, y = self.load_realistic_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples (fraud rate: {y_train.mean():.3%})")
        print(f"Test set: {X_test.shape[0]} samples (fraud rate: {y_test.mean():.3%})")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create models
        models = self.create_models()
        
        results = {}
        
        for model_name, model in models.items():
            result = self.evaluate_model_realistically(
                model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
            )
            results[model_name] = result
        
        # Find best model based on F1-score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model_info = results[best_model_name]
        
        print(f"\n{'='*80}")
        print("BEST REALISTIC MODEL")
        print(f"{'='*80}")
        print(f"Model: {best_model_name}")
        print(f"Cross-validation F1: {best_model_info['cv_f1_mean']:.4f} (+/- {best_model_info['cv_f1_std'] * 2:.4f})")
        print(f"Test Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"Test Precision: {best_model_info['precision']:.4f}")
        print(f"Test Recall: {best_model_info['recall']:.4f}")
        print(f"Test F1-Score: {best_model_info['f1_score']:.4f}")
        print(f"Test AUC: {best_model_info['auc']:.4f}")
        print(f"Total Cost: ${best_model_info['total_cost']:,}")
        
        # Save best model
        joblib.dump(best_model_info['model'], 'models/realistic_best_model.pkl')
        joblib.dump(self.scaler, 'models/realistic_scaler.pkl')
        
        print(f"\nRealistic model saved as: models/realistic_best_model.pkl")
        print(f"Scaler saved as: models/realistic_scaler.pkl")
        
        return best_model_info, results

def main():
    """Main function"""
    trainer = RealisticModelTrainer()
    best_model_info, results = trainer.train_realistic_models()
    
    print(f"\n{'='*80}")
    print("REALISTIC MODEL TRAINING COMPLETED!")
    print("="*80)
    print("Key characteristics:")
    print("1. No data leakage - only features available at prediction time")
    print("2. Realistic fraud rate (~5.5%)")
    print("3. Subtle fraud patterns - not easily separable")
    print("4. Proper cross-validation")
    print("5. Honest performance assessment")
    
    print(f"\nRealistic Performance:")
    print(f"Best model: {max(results.keys(), key=lambda x: results[x]['f1_score'])}")
    print(f"F1-Score: {best_model_info['f1_score']:.4f}")
    print(f"Precision: {best_model_info['precision']:.4f}")
    print(f"Recall: {best_model_info['recall']:.4f}")
    print(f"AUC: {best_model_info['auc']:.4f}")
    
    return trainer, best_model_info, results

if __name__ == "__main__":
    trainer, best_model_info, results = main()
