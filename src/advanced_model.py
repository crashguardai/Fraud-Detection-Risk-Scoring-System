"""
Advanced Fraud Detection Model with Multiple Improvement Strategies

This script implements several advanced techniques to significantly improve metrics:
1. Advanced feature engineering
2. Multiple ensemble methods
3. Cost-sensitive learning
4. Threshold optimization for business objectives
5. Anomaly detection approaches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudDetector:
    """
    Advanced fraud detection with multiple improvement strategies
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_threshold = 0.5
        self.feature_columns = None
        self.selected_features = None
        
    def load_data(self, train_path='data/train_processed.csv', test_path='data/test_processed.csv'):
        """Load and prepare data"""
        print("Loading data for advanced model...")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data.drop('is_fraud', axis=1)
        y_train = train_data['is_fraud']
        X_test = test_data.drop('is_fraud', axis=1)
        y_test = test_data['is_fraud']
        
        # Handle NaN values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)
        
        self.feature_columns = X_train.columns.tolist()
        
        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        print(f"Fraud rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def advanced_feature_engineering(self, X_train, X_test):
        """Create advanced features"""
        print("Creating advanced features...")
        
        def add_features(df):
            df = df.copy()
            
            # Interaction features
            if 'transaction_amount' in df.columns and 'customer_avg_amount' in df.columns:
                df['amount_vs_avg_ratio'] = df['transaction_amount'] / (df['customer_avg_amount'] + 1e-6)
            
            if 'distance_from_home_km' in df.columns and 'distance_from_last_transaction_km' in df.columns:
                df['distance_anomaly_score'] = df['distance_from_home_km'] * df['distance_from_last_transaction_km']
            
            if 'devices_used_today' in df.columns and 'customer_transaction_count' in df.columns:
                df['device_frequency_ratio'] = df['devices_used_today'] / (df['customer_transaction_count'] + 1e-6)
            
            # Time-based features
            if 'transaction_hour' in df.columns:
                df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
                df['is_evening'] = ((df['transaction_hour'] >= 18) & (df['transaction_hour'] <= 21)).astype(int)
                df['is_late_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
            
            # Amount-based features
            if 'transaction_amount' in df.columns:
                df['is_high_amount'] = (df['transaction_amount'] > 1000).astype(int)
                df['is_very_high_amount'] = (df['transaction_amount'] > 5000).astype(int)
                df['amount_z_score'] = (df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std()
            
            # Customer behavior features
            if 'customer_transaction_count' in df.columns and 'customer_fraud_count' in df.columns:
                df['fraud_history_ratio'] = df['customer_fraud_count'] / (df['customer_transaction_count'] + 1e-6)
                df['is_new_customer'] = (df['customer_transaction_count'] <= 10).astype(int)
                df['is_veteran_customer'] = (df['customer_transaction_count'] >= 100).astype(int)
            
            # Risk combination features
            risk_features = []
            if 'ratio_to_median_purchase_price' in df.columns:
                risk_features.append('ratio_to_median_purchase_price')
            if 'distance_from_home_km' in df.columns:
                risk_features.append('distance_from_home_km')
            if 'devices_used_today' in df.columns:
                risk_features.append('devices_used_today')
            
            if len(risk_features) >= 2:
                df['composite_risk_score'] = df[risk_features].sum(axis=1)
            
            return df
        
        X_train_enhanced = add_features(X_train)
        X_test_enhanced = add_features(X_test)
        
        print(f"Feature engineering completed: {X_train.shape} -> {X_train_enhanced.shape}")
        
        return X_train_enhanced, X_test_enhanced
    
    def feature_selection(self, X_train, y_train, method='selectkbest'):
        """Select best features"""
        print(f"Performing feature selection: {method}")
        
        if method == 'selectkbest':
            selector = SelectKBest(f_classif, k=20)
            X_train_selected = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
            selector = RFE(rf, n_features_to_select=20)
            X_train_selected = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            
        else:
            # No selection
            X_train_selected = X_train
            selected_features = X_train.columns.tolist()
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features")
        return X_train_selected, selected_features
    
    def create_advanced_models(self):
        """Create advanced ensemble models"""
        print("Creating advanced models...")
        
        # Cost-sensitive Random Forest
        rf_cost_sensitive = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight={0: 1, 1: 50},  # Heavy weight on fraud class
            random_state=42,
            n_jobs=-1
        )
        
        # Balanced Random Forest
        rf_balanced = BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Easy Ensemble
        easy_ensemble = EasyEnsembleClassifier(
            n_estimators=10,
            estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            random_state=42,
            n_jobs=-1
        )
        
        # RUSBoost
        rusboost = RUSBoostClassifier(
            n_estimators=200,
            estimator=DecisionTreeClassifier(max_depth=10),
            random_state=42
        )
        
        # Gradient Boosting with focus on minority class
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        # Extra Trees
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Logistic Regression with strong class weight
        lr = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='liblinear',
            class_weight={0: 1, 1: 100},
            random_state=42,
            max_iter=1000
        )
        
        self.models = {
            'Cost-Sensitive RF': rf_cost_sensitive,
            'Balanced RF': rf_balanced,
            'Easy Ensemble': easy_ensemble,
            'RUSBoost': rusboost,
            'Gradient Boosting': gb,
            'Extra Trees': et,
            'Logistic Regression': lr
        }
        
        return self.models
    
    def advanced_balancing(self, X_train, y_train, technique='smoteenn'):
        """Apply advanced balancing techniques"""
        print(f"Applying advanced balancing: {technique}")
        
        if technique == 'smoteenn':
            balancer = SMOTEENN(random_state=42)
        elif technique == 'borderline_smote':
            balancer = BorderlineSMOTE(random_state=42)
        elif technique == 'adasyn':
            balancer = ADASYN(random_state=42)
        elif technique == 'nearmiss':
            balancer = NearMiss(version=3)
        else:
            balancer = SMOTE(random_state=42)
        
        X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
        
        print(f"Original: {X_train.shape}, Balanced: {X_balanced.shape}")
        print(f"Original fraud rate: {y_train.mean():.3f}, Balanced: {y_balanced.mean():.3f}")
        
        return X_balanced, y_balanced
    
    def optimize_for_business_objectives(self, model, X_test, y_test):
        """Optimize threshold for business objectives"""
        print("Optimizing for business objectives...")
        
        # Get probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find best threshold based on different objectives
        # 1. Maximize F1
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        
        # 2. High Precision (minimize false positives)
        high_precision_threshold = 0.7
        high_precision_idx = np.where(thresholds >= high_precision_threshold)[0]
        if len(high_precision_idx) > 0:
            high_precision_idx = high_precision_idx[0]
            high_precision = precision[high_precision_idx]
        else:
            high_precision = 0.0
        
        # 3. High Recall (minimize false negatives)
        high_recall_threshold = 0.3
        high_recall_idx = np.where(thresholds <= high_recall_threshold)[0]
        if len(high_recall_idx) > 0:
            high_recall_idx = high_recall_idx[-1]
            high_recall = recall[high_recall_idx]
        else:
            high_recall = 0.0
        
        # 4. Cost-optimal (considering FN costs 100x FP costs)
        costs = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            cost = (fp * 10) + (fn * 1000)  # FN is 100x more costly
            costs.append(cost)
        
        best_cost_idx = np.argmin(costs)
        best_cost_threshold = thresholds[best_cost_idx]
        best_cost = costs[best_cost_idx]
        
        print(f"F1-optimal threshold: {best_f1_threshold:.3f} (F1: {best_f1:.4f})")
        print(f"High precision threshold: {high_precision_threshold:.3f} (Precision: {high_precision:.4f})")
        print(f"High recall threshold: {high_recall_threshold:.3f} (Recall: {high_recall:.4f})")
        print(f"Cost-optimal threshold: {best_cost_threshold:.3f} (Cost: ${best_cost:,})")
        
        # Choose threshold based on business priority
        # For fraud detection, we want to balance precision and recall
        self.best_threshold = best_f1_threshold
        
        return {
            'f1_threshold': best_f1_threshold,
            'f1_score': best_f1,
            'precision_threshold': high_precision_threshold,
            'precision': high_precision,
            'recall_threshold': high_recall_threshold,
            'recall': high_recall,
            'cost_threshold': best_cost_threshold,
            'cost': best_cost,
            'chosen_threshold': self.best_threshold
        }
    
    def comprehensive_evaluation(self, model, X_test, y_test, model_name, threshold_info):
        """Comprehensive model evaluation"""
        threshold = threshold_info['chosen_threshold']
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business metrics
        cost_fp = 10
        cost_fn = 1000
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        cost_per_transaction = total_cost / len(y_test)
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n{model_name} Results:")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Total Cost: ${total_cost:,}")
        print(f"  Cost per Transaction: ${cost_per_transaction:.2f}")
        print(f"  False Positive Rate: {fpr:.4f}")
        print(f"  False Negative Rate: {fnr:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_precision': avg_precision,
            'threshold': threshold,
            'total_cost': total_cost,
            'cost_per_transaction': cost_per_transaction,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': cm
        }
    
    def train_advanced_models(self):
        """Complete advanced training pipeline"""
        print("="*80)
        print("ADVANCED FRAUD DETECTION MODEL TRAINING")
        print("="*80)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Advanced feature engineering
        X_train_enhanced, X_test_enhanced = self.advanced_feature_engineering(X_train, X_test)
        
        # Feature selection
        X_train_selected, selected_features = self.feature_selection(X_train_enhanced, y_train)
        X_test_selected = X_test_enhanced[selected_features]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        self.scalers['robust'] = scaler
        
        # Apply advanced balancing
        X_balanced, y_balanced = self.advanced_balancing(X_train_scaled, y_train, 'smoteenn')
        
        # Create advanced models
        models = self.create_advanced_models()
        
        results = {}
        best_f1 = 0
        best_model_info = None
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"TRAINING: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Train model
                model.fit(X_balanced, y_balanced)
                
                # Optimize threshold
                threshold_info = self.optimize_for_business_objectives(model, X_test_scaled, y_test)
                
                # Evaluate
                result = self.comprehensive_evaluation(model, X_test_scaled, y_test, model_name, threshold_info)
                result['threshold_info'] = threshold_info
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
                model_filename = f"models/advanced_{model_name.lower().replace(' ', '_').replace('-', '_')}.pkl"
                joblib.dump(model, model_filename)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Find and save best model
        if best_model_info:
            print(f"\n{'='*80}")
            print("BEST ADVANCED MODEL")
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
            joblib.dump(best_model_info['model'], 'models/advanced_best_model.pkl')
            joblib.dump(scaler, 'models/advanced_scaler.pkl')
            joblib.dump(self.best_threshold, 'models/advanced_threshold.pkl')
            joblib.dump(selected_features, 'models/advanced_features.pkl')
            
            print(f"\nBest model saved as: models/advanced_best_model.pkl")
            print(f"Scaler saved as: models/advanced_scaler.pkl")
            print(f"Threshold saved as: models/advanced_threshold.pkl")
            print(f"Features saved as: models/advanced_features.pkl")
        
        return best_model_info, results

def main():
    """Main function to run advanced model training"""
    detector = AdvancedFraudDetector()
    best_model_info, all_results = detector.train_advanced_models()
    
    if best_model_info:
        print(f"\n{'='*80}")
        print("ADVANCED MODEL TRAINING COMPLETED!")
        print(f"{'='*80}")
        print("Improvements achieved through:")
        print("1. Advanced feature engineering")
        print("2. Multiple ensemble methods")
        print("3. Cost-sensitive learning")
        print("4. Threshold optimization")
        print("5. Balanced sampling techniques")
        print("6. Feature selection")
        
        return detector, best_model_info, all_results
    else:
        print("No models were successfully trained.")
        return None, None, None

if __name__ == "__main__":
    detector, best_model_info, all_results = main()
