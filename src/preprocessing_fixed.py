"""
Data Preprocessing and Feature Engineering for Fraud Detection

This module handles all data preprocessing steps including:
- Missing value imputation
- Feature encoding
- Feature scaling
- Feature engineering
- Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPreprocessor:
    """
    A comprehensive preprocessor for fraud detection data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_columns = None
        self.target_column = 'is_fraud'
        
    def load_data(self, filepath):
        """
        Load and prepare the dataset
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Convert datetime columns
        if 'transaction_time' in df.columns:
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
            
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with handled missing values
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        print("Missing values summary:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({missing_percentages[col]:.2f}%)")
        
        # Handle missing values based on data type
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    # For categorical columns, use mode
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                else:
                    # For numerical columns, use median
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
        
        print("Missing values handled successfully.")
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        print("Performing feature engineering...")
        
        # Create time-based features if transaction_time exists
        if 'transaction_time' in df.columns:
            df['transaction_day_of_week'] = df['transaction_time'].dt.dayofweek
            df['transaction_month'] = df['transaction_time'].dt.month
            df['is_month_end'] = df['transaction_time'].dt.is_month_end.astype(int)
            df['is_month_start'] = df['transaction_time'].dt.is_month_start.astype(int)
        
        # Create amount-based features
        if 'transaction_amount' in df.columns:
            # Log transformation for amount (helps with skewed distributions)
            df['log_transaction_amount'] = np.log1p(df['transaction_amount'])
            
            # Amount bins
            df['amount_category'] = pd.cut(df['transaction_amount'], 
                                         bins=[0, 10, 50, 100, 500, float('inf')],
                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Create distance-based features
        if 'distance_from_home_km' in df.columns and 'distance_from_last_transaction_km' in df.columns:
            # Total distance traveled
            df['total_distance_km'] = df['distance_from_home_km'] + df['distance_from_last_transaction_km']
            
            # Distance ratio
            df['distance_ratio'] = df['distance_from_last_transaction_km'] / (df['distance_from_home_km'] + 1e-6)
        
        # Create customer behavior features
        if 'customer_transaction_count' in df.columns and 'customer_fraud_count' in df.columns:
            # Customer fraud rate
            df['customer_fraud_rate'] = df['customer_fraud_count'] / (df['customer_transaction_count'] + 1e-6)
            
            # Customer experience level
            df['customer_experience'] = pd.cut(df['customer_transaction_count'],
                                             bins=[0, 5, 20, 50, float('inf')],
                                             labels=['New', 'Regular', 'Experienced', 'Very Experienced'])
        
        # Risk scoring features
        if 'ratio_to_median_purchase_price' in df.columns:
            # Unusual spending indicator
            df['is_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 3).astype(int)
            df['is_very_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 5).astype(int)
        
        # Device usage risk
        if 'devices_used_today' in df.columns:
            df['is_multiple_devices'] = (df['devices_used_today'] > 3).astype(int)
        
        # Time-based risk features
        if 'transaction_hour' in df.columns:
            # Categorize hours into risk periods
            df['time_risk_category'] = pd.cut(df['transaction_hour'],
                                            bins=[0, 6, 12, 18, 24],
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                            ordered=False)
        
        print(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with encoded categorical features
        """
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        if self.target_column in categorical_columns:
            categorical_columns.remove(self.target_column)
        
        # Remove datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in datetime_columns]
        
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Use label encoding for ordinal categories and one-hot for nominal
        ordinal_mappings = {
            'amount_category': {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
            'customer_experience': {'New': 0, 'Regular': 1, 'Experienced': 2, 'Very Experienced': 3},
            'time_risk_category': {'Night': 2, 'Morning': 0, 'Afternoon': 1, 'Evening': 1}  # Night is highest risk
        }
        
        for col in categorical_columns:
            if col in ordinal_mappings:
                # Use label encoding for ordinal features
                if col not in self.label_encoders:
                    self.label_encoders[col] = ordinal_mappings[col]
                df[col] = df[col].map(self.label_encoders[col])
            else:
                # Use one-hot encoding for nominal features
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        print("Categorical features encoded successfully.")
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (features DataFrame, target Series)
        """
        print("Preparing features for modeling...")
        
        # Separate features and target
        if self.target_column in df.columns:
            y = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Remove non-numeric columns that shouldn't be used for modeling
        columns_to_drop = ['transaction_id', 'transaction_time', 'customer_id']
        for col in columns_to_drop:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        
        # Keep only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_columns]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Features prepared. Number of features: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: (scaled X_train, scaled X_test)
        """
        print("Scaling features...")
        
        # Handle any remaining NaN values
        print(f"Checking for NaN values before scaling...")
        train_nan_count = X_train.isnull().sum().sum()
        test_nan_count = X_test.isnull().sum().sum()
        
        if train_nan_count > 0:
            print(f"Found {train_nan_count} NaN values in training data, filling with median...")
            X_train = X_train.fillna(X_train.median())
        
        if test_nan_count > 0:
            print(f"Found {test_nan_count} NaN values in test data, filling with median...")
            X_test = X_test.fillna(X_train.median())  # Use training median for test data
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        print("Features scaled successfully.")
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify the split
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data with test_size={test_size}...")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        print(f"  Training fraud rate: {y_train.mean():.3f}")
        print(f"  Test fraud rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath, test_size=0.2):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to the raw data file
            test_size: Proportion of test set
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        print("Starting preprocessing pipeline...")
        print("=" * 50)
        
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("=" * 50)
        print("Preprocessing pipeline completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """
        Save the preprocessor object
        
        Args:
            filepath: Path to save the preprocessor
        """
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """
        Load a saved preprocessor
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            FraudDetectionPreprocessor: Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor

def main():
    """
    Main function to run preprocessing
    """
    # Initialize preprocessor
    preprocessor = FraudDetectionPreprocessor()
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        'data/fraud_data.csv',
        test_size=0.2
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Save processed data for modeling
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/train_processed.csv', index=False)
    test_data.to_csv('data/test_processed.csv', index=False)
    
    print("Processed data saved to 'data/train_processed.csv' and 'data/test_processed.csv'")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()
