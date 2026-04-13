"""
Data Generation for Fraud Detection System

This script generates a realistic sample dataset for fraud detection
with various features that are commonly found in financial transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02):
    """
    Generate a comprehensive fraud detection dataset
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Proportion of fraudulent transactions (default 2%)
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    
    print(f"Generating {n_samples} samples with {fraud_ratio*100}% fraud rate...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Calculate number of fraudulent transactions
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Generate base features
    data = []
    
    # Transaction amounts - fraud transactions tend to have different patterns
    legitimate_amounts = np.random.lognormal(mean=3, sigma=1.5, size=n_legitimate)
    legitimate_amounts = np.clip(legitimate_amounts, 1, 10000)
    
    fraud_amounts = np.concatenate([
        np.random.lognormal(mean=4, sigma=2, size=n_fraud//2),  # High value fraud
        np.random.lognormal(mean=2.5, sigma=0.8, size=n_fraud - n_fraud//2)  # Small value fraud
    ])
    fraud_amounts = np.clip(fraud_amounts, 1, 50000)
    
    # Time-based features
    start_date = datetime.now() - timedelta(days=90)
    
    # Generate legitimate transactions
    for i in range(n_legitimate):
        transaction_time = start_date + timedelta(
            minutes=np.random.randint(0, 90*24*60)
        )
        
        # Customer behavior patterns
        customer_id = np.random.randint(1, 1000)
        customer_age = np.random.randint(18, 80)
        customer_tenure = np.random.randint(1, 365)  # days as customer
        
        # Transaction patterns
        merchant_category = np.random.choice(['retail', 'food', 'gas', 'online', 'travel', 'entertainment'])
        transaction_hour = transaction_time.hour
        
        # Location features
        distance_from_home = np.random.exponential(scale=50)  # km
        distance_from_last_transaction = np.random.exponential(scale=10)  # km
        
        # Device features
        devices_used_today = np.random.randint(1, 5)
        is_mobile = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # Risk indicators (low for legitimate)
        ratio_to_median_purchase_price = np.random.normal(1.0, 0.3)
        ratio_to_median_purchase_price = max(0.1, ratio_to_median_purchase_price)
        
        data.append({
            'transaction_id': i + 1,
            'transaction_time': transaction_time,
            'transaction_amount': legitimate_amounts[i],
            'customer_id': customer_id,
            'customer_age': customer_age,
            'customer_tenure_days': customer_tenure,
            'merchant_category': merchant_category,
            'transaction_hour': transaction_hour,
            'distance_from_home_km': distance_from_home,
            'distance_from_last_transaction_km': distance_from_last_transaction,
            'devices_used_today': devices_used_today,
            'is_mobile_transaction': is_mobile,
            'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
            'is_fraud': 0
        })
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        transaction_time = start_date + timedelta(
            minutes=np.random.randint(0, 90*24*60)
        )
        
        # Fraud patterns
        customer_id = np.random.randint(1, 1000)
        customer_age = np.random.randint(18, 80)
        customer_tenure = np.random.randint(1, 365)
        
        # Fraudulent transactions often occur in specific categories
        merchant_category = np.random.choice(['online', 'travel', 'retail'], p=[0.5, 0.3, 0.2])
        transaction_hour = np.random.choice([2, 3, 4, 22, 23, 0, 1], p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1])
        
        # Fraud indicators
        distance_from_home = np.random.exponential(scale=200)  # Usually far from home
        distance_from_last_transaction = np.random.exponential(scale=100)  # Unusual distances
        
        devices_used_today = np.random.randint(3, 10)  # Multiple devices
        is_mobile = np.random.choice([0, 1], p=[0.2, 0.8])  # More likely mobile
        
        # High ratio to median purchase (unusual spending)
        ratio_to_median_purchase_price = np.random.choice([
            np.random.normal(5.0, 2.0),  # Very high
            np.random.normal(0.2, 0.1)   # Very low (testing)
        ], p=[0.7, 0.3])
        ratio_to_median_purchase_price = max(0.01, ratio_to_median_purchase_price)
        
        data.append({
            'transaction_id': n_legitimate + i + 1,
            'transaction_time': transaction_time,
            'transaction_amount': fraud_amounts[i],
            'customer_id': customer_id,
            'customer_age': customer_age,
            'customer_tenure_days': customer_tenure,
            'merchant_category': merchant_category,
            'transaction_hour': transaction_hour,
            'distance_from_home_km': distance_from_home,
            'distance_from_last_transaction_km': distance_from_last_transaction,
            'devices_used_today': devices_used_today,
            'is_mobile_transaction': is_mobile,
            'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
            'is_fraud': 1
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some derived features
    df['is_weekend'] = df['transaction_time'].dt.weekday >= 5
    df['is_night_time'] = df['transaction_hour'].between(22, 6)
    
    # Customer-level aggregations
    customer_stats = df.groupby('customer_id').agg({
        'transaction_amount': ['mean', 'std', 'count'],
        'is_fraud': 'sum'
    }).reset_index()
    
    customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 
                              'customer_transaction_count', 'customer_fraud_count']
    
    # Merge customer stats back
    df = df.merge(customer_stats, on='customer_id', how='left')
    
    # Fill NaN values for customers with only one transaction
    df['customer_std_amount'] = df['customer_std_amount'].fillna(0)
    
    # Reorder columns for better readability
    column_order = [
        'transaction_id', 'transaction_time', 'transaction_amount',
        'customer_id', 'customer_age', 'customer_tenure_days',
        'merchant_category', 'transaction_hour', 'is_weekend', 'is_night_time',
        'distance_from_home_km', 'distance_from_last_transaction_km',
        'devices_used_today', 'is_mobile_transaction',
        'ratio_to_median_purchase_price',
        'customer_avg_amount', 'customer_std_amount', 'customer_transaction_count',
        'customer_fraud_count', 'is_fraud'
    ]
    
    df = df[column_order]
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
    print(f"Legitimate transactions: {(df['is_fraud'] == 0).sum()} ({(df['is_fraud'] == 0).mean():.2%})")
    
    return df

def save_dataset(df, filepath='data/fraud_data.csv'):
    """Save the dataset to CSV file"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    # Generate the dataset
    df = generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02)
    
    # Save to file
    save_dataset(df)
    
    # Display basic statistics
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nSample of the data:")
    print(df.head())
    
    print("\nFraud distribution:")
    print(df['is_fraud'].value_counts(normalize=True))
