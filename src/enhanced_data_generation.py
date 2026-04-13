"""
Enhanced Data Generation for Fraud Detection

This script creates a much better dataset with:
1. More realistic fraud patterns
2. Better class balance (5-10% fraud rate)
3. Clearer fraud indicators
4. More diverse scenarios
5. Better feature distributions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class EnhancedFraudDataGenerator:
    """
    Enhanced fraud data generator with realistic patterns
    """
    
    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        
    def create_customer_profiles(self, n_customers=2000):
        """Create realistic customer profiles"""
        print(f"Creating {n_customers} customer profiles...")
        
        customers = []
        customer_id = 1
        
        # Create different customer segments
        segments = ['low_risk', 'medium_risk', 'high_risk', 'new_customer', 'vip']
        segment_weights = [0.4, 0.3, 0.1, 0.15, 0.05]
        
        for i in range(n_customers):
            segment = np.random.choice(segments, p=segment_weights)
            
            if segment == 'low_risk':
                age = np.random.normal(45, 12)
                tenure = np.random.exponential(scale=1000) + 100
                income = np.random.lognormal(mean=10.5, sigma=0.5)
                risk_score = np.random.uniform(0.1, 0.3)
                
            elif segment == 'medium_risk':
                age = np.random.normal(35, 10)
                tenure = np.random.exponential(scale=500) + 50
                income = np.random.lognormal(mean=10.2, sigma=0.6)
                risk_score = np.random.uniform(0.3, 0.6)
                
            elif segment == 'high_risk':
                age = np.random.normal(30, 8)
                tenure = np.random.exponential(scale=200) + 20
                income = np.random.lognormal(mean=9.8, sigma=0.7)
                risk_score = np.random.uniform(0.6, 0.9)
                
            elif segment == 'new_customer':
                age = np.random.normal(32, 12)
                tenure = np.random.uniform(1, 90)
                income = np.random.lognormal(mean=10.0, sigma=0.8)
                risk_score = np.random.uniform(0.4, 0.8)
                
            else:  # VIP
                age = np.random.normal(50, 10)
                tenure = np.random.exponential(scale=2000) + 500
                income = np.random.lognormal(mean=11.2, sigma=0.4)
                risk_score = np.random.uniform(0.2, 0.4)
            
            # Clamp values to realistic ranges
            age = max(18, min(80, age))
            tenure = max(1, min(3650, tenure))
            income = max(20000, min(500000, income))
            
            customers.append({
                'customer_id': customer_id,
                'age': int(age),
                'tenure_days': int(tenure),
                'income': income,
                'segment': segment,
                'risk_score': risk_score,
                'avg_transaction_amount': np.random.lognormal(mean=3.5, sigma=1.0),
                'transaction_frequency': np.random.poisson(10),
                'preferred_merchant_types': np.random.choice(['retail', 'online', 'food', 'gas'], size=np.random.randint(1, 3)),
                'typical_locations': np.random.choice(['home', 'work', 'travel'], size=np.random.randint(1, 3))
            })
            
            customer_id += 1
        
        return pd.DataFrame(customers)
    
    def generate_legitimate_transactions(self, customers, n_transactions=20000):
        """Generate realistic legitimate transactions"""
        print(f"Generating {n_transactions} legitimate transactions...")
        
        transactions = []
        
        for i in range(n_transactions):
            # Select customer
            customer = customers.iloc[np.random.randint(0, len(customers))]
            
            # Time generation (more realistic patterns)
            base_time = datetime.now() - timedelta(days=np.random.randint(0, 180))
            
            # Add time based on customer behavior
            if customer['segment'] == 'vip':
                # VIPs shop more during business hours
                hour = np.random.choice([9, 10, 11, 14, 15, 16, 17], p=[0.15, 0.15, 0.1, 0.15, 0.15, 0.2, 0.1])
            else:
                # Normal distribution with peaks during lunch and after work
                hour_probs = [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.05, 0.08, 0.12, 0.10, 0.08,  # 0-11
                              0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]  # 12-23
                # Normalize probabilities
                hour_probs = [p/sum(hour_probs) for p in hour_probs]
                hour = np.random.choice(range(24), p=hour_probs)
            
            transaction_time = base_time.replace(hour=hour)
            
            # Amount generation (realistic patterns)
            if customer['segment'] == 'vip':
                amount = np.random.lognormal(mean=5.0, sigma=1.2)
            elif customer['segment'] == 'high_risk':
                amount = np.random.lognormal(mean=3.8, sigma=1.0)
            else:
                amount = np.random.lognormal(mean=4.0, sigma=1.0)
            
            amount = max(1, min(10000, amount))
            
            # Merchant category
            merchant_categories = ['retail', 'online', 'food', 'gas', 'travel', 'entertainment', 'healthcare']
            if len(customer['preferred_merchant_types']) > 0:
                # Higher probability for preferred types
                probs = [0.3 if cat in customer['preferred_merchant_types'] else 0.1 for cat in merchant_categories]
                probs = [p/sum(probs) for p in probs]
                merchant_category = np.random.choice(merchant_categories, p=probs)
            else:
                merchant_category = np.random.choice(merchant_categories)
            
            # Location patterns
            if 'home' in customer['typical_locations']:
                distance_home = np.random.exponential(scale=5)  # Usually close to home
            else:
                distance_home = np.random.exponential(scale=20)  # More varied
            
            if 'work' in customer['typical_locations']:
                distance_last = np.random.exponential(scale=10)
            else:
                distance_last = np.random.exponential(scale=15)
            
            # Device usage
            devices_today = np.random.poisson(2) + 1
            is_mobile = np.random.choice([0, 1], p=[0.4, 0.6])
            
            # Ratio to median (close to 1 for legitimate)
            ratio_to_median = np.random.normal(1.0, 0.3)
            ratio_to_median = max(0.1, min(5.0, ratio_to_median))
            
            transactions.append({
                'transaction_id': i + 1,
                'transaction_time': transaction_time,
                'transaction_amount': amount,
                'customer_id': customer['customer_id'],
                'customer_age': customer['age'],
                'customer_tenure_days': customer['tenure_days'],
                'merchant_category': merchant_category,
                'transaction_hour': hour,
                'distance_from_home_km': distance_home,
                'distance_from_last_transaction_km': distance_last,
                'devices_used_today': devices_today,
                'is_mobile_transaction': bool(is_mobile),
                'ratio_to_median_purchase_price': ratio_to_median,
                'customer_avg_amount': customer['avg_transaction_amount'],
                'customer_std_amount': customer['avg_transaction_amount'] * 0.3,  # Estimated std
                'customer_transaction_count': customer['transaction_frequency'] * 30,  # Monthly estimate
                'customer_fraud_count': 0,
                'is_fraud': 0
            })
        
        return pd.DataFrame(transactions)
    
    def generate_fraud_transactions(self, customers, n_fraud=2000):
        """Generate realistic fraud transactions with clear patterns"""
        print(f"Generating {n_fraud} fraud transactions...")
        
        transactions = []
        fraud_types = ['account_takeover', 'card_theft', 'identity_theft', 'friendly_fraud', 'merchant_fraud']
        fraud_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for i in range(n_fraud):
            fraud_type = np.random.choice(fraud_types, p=fraud_weights)
            
            # Select customer (higher probability for high-risk customers)
            high_risk_mask = customers['risk_score'] > 0.6
            if np.random.random() < 0.7 and high_risk_mask.any():
                customer = customers[high_risk_mask].iloc[np.random.randint(0, high_risk_mask.sum())]
            else:
                customer = customers.iloc[np.random.randint(0, len(customers))]
            
            # Time patterns for fraud
            if fraud_type == 'account_takeover':
                # Fraudsters often work at odd hours
                hour = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.12, 0.13])
            elif fraud_type == 'card_theft':
                # Can happen any time, often quick transactions
                hour = np.random.randint(0, 24)
            elif fraud_type == 'identity_theft':
                # Often during business hours to blend in
                hour = np.random.choice([9, 10, 11, 14, 15, 16, 17], p=[0.15, 0.15, 0.1, 0.15, 0.15, 0.2, 0.1])
            else:
                hour = np.random.randint(0, 24)
            
            base_time = datetime.now() - timedelta(days=np.random.randint(0, 90))
            transaction_time = base_time.replace(hour=hour)
            
            # Amount patterns for fraud
            if fraud_type == 'account_takeover':
                # Often high-value transactions
                amount = np.random.lognormal(mean=6.0, sigma=1.5)
            elif fraud_type == 'card_theft':
                # Quick multiple transactions
                amount = np.random.lognormal(mean=4.5, sigma=1.2)
            elif fraud_type == 'identity_theft':
                # Medium to high amounts
                amount = np.random.lognormal(mean=5.5, sigma=1.3)
            elif fraud_type == 'friendly_fraud':
                # Often specific amounts
                amount = np.random.choice([99.99, 199.99, 499.99, 999.99])
            else:  # merchant_fraud
                # Small frequent transactions
                amount = np.random.lognormal(mean=3.5, sigma=0.8)
            
            amount = max(1, min(50000, amount))
            
            # Merchant category (fraudsters prefer certain types)
            if fraud_type == 'account_takeover':
                merchant_category = np.random.choice(['online', 'travel', 'electronics'], p=[0.5, 0.3, 0.2])
            elif fraud_type == 'card_theft':
                merchant_category = np.random.choice(['online', 'retail', 'gas'], p=[0.4, 0.4, 0.2])
            else:
                merchant_category = np.random.choice(['online', 'retail', 'food', 'gas'])
            
            # Geographic patterns (fraud often occurs far from normal locations)
            distance_home = np.random.exponential(scale=100)  # Much farther from home
            distance_last = np.random.exponential(scale=50)   # Far from last transaction
            
            # Device patterns
            if fraud_type == 'account_takeover':
                devices_today = np.random.randint(3, 10)  # Multiple devices
                is_mobile = True  # Usually mobile
            else:
                devices_today = np.random.randint(1, 5)
                is_mobile = np.random.choice([0, 1], p=[0.3, 0.7])
            
            # Spending patterns (unusual ratios)
            if fraud_type == 'account_takeover':
                ratio_to_median = np.random.uniform(5.0, 20.0)  # Very unusual
            elif fraud_type == 'friendly_fraud':
                ratio_to_median = np.random.uniform(2.0, 8.0)   # Unusual
            else:
                ratio_to_median = np.random.uniform(3.0, 15.0)  # Unusual
            
            transactions.append({
                'transaction_id': len(transactions) + 1,
                'transaction_time': transaction_time,
                'transaction_amount': amount,
                'customer_id': customer['customer_id'],
                'customer_age': customer['age'],
                'customer_tenure_days': customer['tenure_days'],
                'merchant_category': merchant_category,
                'transaction_hour': hour,
                'distance_from_home_km': distance_home,
                'distance_from_last_transaction_km': distance_last,
                'devices_used_today': devices_today,
                'is_mobile_transaction': bool(is_mobile),
                'ratio_to_median_purchase_price': ratio_to_median,
                'customer_avg_amount': customer['avg_transaction_amount'],
                'customer_std_amount': customer['avg_transaction_amount'] * 0.3,
                'customer_transaction_count': customer['transaction_frequency'] * 30,
                'customer_fraud_count': 0,  # Will be updated later
                'is_fraud': 1,
                'fraud_type': fraud_type
            })
        
        return pd.DataFrame(transactions)
    
    def add_customer_aggregations(self, df, customers):
        """Add customer-level aggregations"""
        print("Adding customer-level aggregations...")
        
        # Calculate customer statistics
        customer_stats = df.groupby('customer_id').agg({
            'transaction_amount': ['mean', 'std', 'count', 'sum'],
            'is_fraud': ['sum', 'mean']
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 
                                 'customer_transaction_count', 'customer_total_amount',
                                 'customer_fraud_count', 'customer_fraud_rate']
        
        # Handle NaN std
        customer_stats['customer_std_amount'] = customer_stats['customer_std_amount'].fillna(customer_stats['customer_avg_amount'] * 0.2)
        
        # Merge back
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Add customer profile info
        df = df.merge(customers[['customer_id', 'segment', 'risk_score']], on='customer_id', how='left')
        
        return df
    
    def create_enhanced_dataset(self, n_customers=2000, fraud_rate=0.08):
        """Create enhanced dataset with better fraud patterns"""
        print("="*60)
        print("CREATING ENHANCED FRAUD DETECTION DATASET")
        print("="*60)
        
        # Create customer profiles
        customers = self.create_customer_profiles(n_customers)
        
        # Calculate transaction counts
        total_transactions = int(50000)  # Larger dataset
        n_fraud = int(total_transactions * fraud_rate)
        n_legitimate = total_transactions - n_fraud
        
        # Generate transactions
        legitimate_df = self.generate_legitimate_transactions(customers, n_legitimate)
        fraud_df = self.generate_fraud_transactions(customers, n_fraud)
        
        # Combine datasets
        df = pd.concat([legitimate_df, fraud_df], ignore_index=True)
        
        # Add customer aggregations
        # Remove existing customer stats columns first
        existing_cols = ['customer_avg_amount', 'customer_std_amount', 'customer_transaction_count', 'customer_fraud_count']
        for col in existing_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        df = self.add_customer_aggregations(df, customers)
        
        # Add time-based features
        df['transaction_day_of_week'] = df['transaction_time'].dt.dayofweek
        df['transaction_month'] = df['transaction_time'].dt.month
        df['is_weekend'] = df['transaction_day_of_week'] >= 5
        df['is_night_time'] = df['transaction_hour'].between(22, 6)
        df['is_month_end'] = df['transaction_time'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['transaction_time'].dt.is_month_start.astype(int)
        
        # Add derived features
        df['log_transaction_amount'] = np.log1p(df['transaction_amount'])
        
        # Distance features
        df['total_distance_km'] = df['distance_from_home_km'] + df['distance_from_last_transaction_km']
        df['distance_ratio'] = df['distance_from_last_transaction_km'] / (df['distance_from_home_km'] + 1e-6)
        
        # Customer behavior features
        df['customer_fraud_rate'] = df['customer_fraud_count'] / (df['customer_transaction_count'] + 1e-6)
        df['customer_experience_level'] = pd.cut(df['customer_tenure_days'],
                                                 bins=[0, 30, 180, 730, float('inf')],
                                                 labels=['New', 'Regular', 'Experienced', 'Very Experienced'])
        
        # Risk indicators
        df['is_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 3).astype(int)
        df['is_very_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 5).astype(int)
        df['is_multiple_devices'] = (df['devices_used_today'] > 3).astype(int)
        df['is_far_from_home'] = (df['distance_from_home_km'] > 50).astype(int)
        df['is_late_night'] = df['transaction_hour'].between(0, 6).astype(int)
        
        # Time risk categories
        df['time_risk_category'] = pd.cut(df['transaction_hour'],
                                        bins=[0, 6, 12, 18, 24],
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                        ordered=False)
        
        # Amount categories
        df['amount_category'] = pd.cut(df['transaction_amount'],
                                      bins=[0, 25, 100, 500, 2000, float('inf')],
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Sort and reset index
        df = df.sort_values('transaction_time').reset_index(drop=True)
        
        # Select final columns
        final_columns = [
            'transaction_id', 'transaction_time', 'transaction_amount',
            'customer_id', 'customer_age', 'customer_tenure_days',
            'merchant_category', 'transaction_hour', 'is_weekend', 'is_night_time',
            'distance_from_home_km', 'distance_from_last_transaction_km',
            'devices_used_today', 'is_mobile_transaction',
            'ratio_to_median_purchase_price',
            'customer_avg_amount', 'customer_std_amount',
            'customer_transaction_count', 'customer_fraud_count',
            'transaction_day_of_week', 'transaction_month', 'is_month_end', 'is_month_start',
            'log_transaction_amount', 'total_distance_km', 'distance_ratio',
            'customer_fraud_rate', 'is_unusual_spending', 'is_very_unusual_spending',
            'is_multiple_devices', 'is_far_from_home', 'is_late_night',
            'time_risk_category', 'amount_category', 'segment', 'risk_score',
            'is_fraud'
        ]
        
        df = df[final_columns]
        
        # Drop unnecessary columns for modeling
        df_model = df.drop(['transaction_id', 'transaction_time', 'customer_id'], axis=1)
        
        print(f"\nDataset created successfully!")
        print(f"Total transactions: {len(df)}")
        print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        print(f"Legitimate transactions: {(df['is_fraud'] == 0).sum()} ({(df['is_fraud'] == 0).mean():.2%})")
        print(f"Number of features: {len(df_model.columns)}")
        
        return df, df_model
    
    def save_dataset(self, df, df_model, filepath='data/enhanced_fraud_data.csv'):
        """Save the enhanced dataset"""
        df.to_csv(filepath, index=False)
        print(f"Enhanced dataset saved to {filepath}")
        
        # Also save model-ready version
        model_filepath = filepath.replace('.csv', '_model_ready.csv')
        df_model.to_csv(model_filepath, index=False)
        print(f"Model-ready dataset saved to {model_filepath}")

def main():
    """Main function to generate enhanced dataset"""
    generator = EnhancedFraudDataGenerator()
    
    # Create enhanced dataset with 8% fraud rate (much better than 2%)
    df, df_model = generator.create_enhanced_dataset(n_customers=2000, fraud_rate=0.08)
    
    # Save datasets
    generator.save_dataset(df, df_model)
    
    # Display some statistics
    print(f"\n{'='*60}")
    print("ENHANCED DATASET STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nFraud by Type:")
    if 'fraud_type' in df.columns:
        fraud_by_type = df[df['is_fraud'] == 1]['fraud_type'].value_counts()
        for fraud_type, count in fraud_by_type.items():
            print(f"  {fraud_type}: {count} ({count/df['is_fraud'].sum():.1%})")
    
    print(f"\nFraud by Customer Segment:")
    fraud_by_segment = df[df['is_fraud'] == 1]['segment'].value_counts()
    for segment, count in fraud_by_segment.items():
        print(f"  {segment}: {count} ({count/df['is_fraud'].sum():.1%})")
    
    print(f"\nFraud by Merchant Category:")
    fraud_by_merchant = df[df['is_fraud'] == 1]['merchant_category'].value_counts()
    for merchant, count in fraud_by_merchant.items():
        print(f"  {merchant}: {count} ({count/df['is_fraud'].sum():.1%})")
    
    print(f"\nKey Fraud Indicators in Fraudulent Transactions:")
    print(f"  Unusual spending (>3x): {df[df['is_fraud'] == 1]['is_unusual_spending'].mean():.1%}")
    print(f"  Multiple devices (>3): {df[df['is_fraud'] == 1]['is_multiple_devices'].mean():.1%}")
    print(f"  Far from home (>50km): {df[df['is_fraud'] == 1]['is_far_from_home'].mean():.1%}")
    print(f"  Late night (0-6am): {df[df['is_fraud'] == 1]['is_late_night'].mean():.1%}")
    
    return df, df_model

if __name__ == "__main__":
    df, df_model = main()
