"""
Model Training Script for Fraud Detection

This script provides a standalone way to train and evaluate models
without using Jupyter notebooks.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModelTrainer:
    """
    Complete model training pipeline for fraud detection
    """
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, train_path='data/train_processed.csv', 
                  test_path='data/test_processed.csv'):
        """
        Load preprocessed data
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading preprocessed data...")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data.drop('is_fraud', axis=1)
        y_train = train_data['is_fraud']
        X_test = test_data.drop('is_fraud', axis=1)
        y_test = test_data['is_fraud']
        
        # Handle any remaining NaN values
        print("Checking for NaN values...")
        train_nan_count = X_train.isnull().sum().sum()
        test_nan_count = X_test.isnull().sum().sum()
        train_y_nan = y_train.isnull().sum()
        test_y_nan = y_test.isnull().sum()
        
        if train_nan_count > 0:
            print(f"Found {train_nan_count} NaN values in training data, filling with median...")
            X_train = X_train.fillna(X_train.median())
        
        if test_nan_count > 0:
            print(f"Found {test_nan_count} NaN values in test data, filling with median...")
            X_test = X_test.fillna(X_train.median())
        
        if train_y_nan > 0:
            print(f"Found {train_y_nan} NaN values in training target, filling with 0...")
            y_train = y_train.fillna(0)
        
        if test_y_nan > 0:
            print(f"Found {test_y_nan} NaN values in test target, filling with 0...")
            y_test = y_test.fillna(0)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training fraud rate: {y_train.mean():.3f}")
        print(f"Test fraud rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize machine learning models
        
        Returns:
            dict: Dictionary of initialized models
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        print("Models initialized:")
        for name, model in models.items():
            print(f"  - {name}")
        
        return models
    
    def train_models(self, models, X_train, y_train):
        """
        Train all models
        
        Args:
            models: Dictionary of models to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\nTraining models...")
        
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  {name} trained successfully.")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            dict: Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation results for all models
        """
        print("\nEvaluating models...")
        
        evaluation_results = {}
        for name, model in self.models.items():
            evaluation_results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def compare_models(self):
        """
        Compare all models and create comparison table
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Evaluate models first.")
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC': results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        # Display table
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best model based on a specific metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_model, best_score)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available.")
        
        best_model_name = max(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x][metric])
        best_score = self.evaluation_results[best_model_name][metric]
        best_model = self.models[best_model_name]
        
        print(f"\nBest model based on {metric}: {best_model_name} (Score: {best_score:.4f})")
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        return best_model_name, best_model, best_score
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Random Forest
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            RandomForestClassifier: Tuned model
        """
        print("\nPerforming hyperparameter tuning for Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize Grid Search
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and score
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self, X_train):
        """
        Analyze feature importance for Random Forest models
        
        Args:
            X_train: Training features (for feature names)
        """
        print("\nFeature Importance Analysis:")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 10 Most Important Features ({name}):")
                print(feature_importance.head(10))
    
    def cross_validation(self, X_train, y_train):
        """
        Perform cross-validation for robust evaluation
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("\nPerforming 5-fold cross-validation...")
        
        from sklearn.model_selection import StratifiedKFold
        
        # Initialize models for cross-validation
        cv_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        }
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in cv_models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            print(f"{model_name} CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def save_models(self, save_dir='models'):
        """
        Save all models and evaluation results
        
        Args:
            save_dir: Directory to save models
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            print(f"Model '{name}' saved to {filepath}")
        
        # Save the best model separately
        if self.best_model is not None:
            best_filepath = os.path.join(save_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_filepath)
            print(f"Best model '{self.best_model_name}' saved to {best_filepath}")
        
        # Save evaluation results
        results_to_save = {}
        for model_name, results in self.evaluation_results.items():
            results_to_save[model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'auc': results['auc']
            }
        
        results_filepath = os.path.join(save_dir, 'model_evaluation_results.json')
        with open(results_filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Evaluation results saved to {results_filepath}")
    
    def run_complete_pipeline(self):
        """
        Run the complete model training pipeline
        """
        print("="*60)
        print("FRAUD DETECTION MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Initialize models
        models = self.initialize_models()
        
        # Train models
        self.train_models(models, X_train, y_train)
        
        # Cross-validation
        self.cross_validation(X_train, y_train)
        
        # Evaluate models
        self.evaluate_all_models(X_test, y_test)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Feature importance analysis
        self.feature_importance_analysis(X_train)
        
        # Get best model
        best_name, best_model, best_score = self.get_best_model()
        
        # Hyperparameter tuning (optional - can be computationally expensive)
        try:
            tuned_model = self.hyperparameter_tuning(X_train, y_train)
            
            # Evaluate tuned model
            tuned_results = self.evaluate_model(tuned_model, X_test, y_test, 'Tuned Random Forest')
            
            # Add tuned model to results
            self.models['Tuned Random Forest'] = tuned_model
            self.evaluation_results['Tuned Random Forest'] = tuned_results
            
            # Compare again
            print("\nFinal Model Comparison (including tuned model):")
            final_comparison = self.compare_models()
            
            # Update best model if tuned model is better
            if tuned_results['f1_score'] > best_score:
                best_name, best_model, best_score = 'Tuned Random Forest', tuned_model, tuned_results['f1_score']
                self.best_model = best_model
                self.best_model_name = best_name
                
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            print("Continuing with original models...")
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {self.best_model_name}")
        print(f"Best F1-Score: {best_score:.4f}")
        
        return self.best_model, self.best_model_name

def main():
    """
    Main function to run model training
    """
    trainer = FraudDetectionModelTrainer()
    best_model, best_model_name = trainer.run_complete_pipeline()
    
    return trainer, best_model, best_model_name

if __name__ == "__main__":
    trainer, best_model, best_model_name = main()
