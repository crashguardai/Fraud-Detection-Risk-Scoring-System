"""
Create Truly Realistic Dataset with No Data Leakage

This creates a dataset that mimics real-world fraud detection challenges.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class RealisticDataGenerator:
    """
    Generate realistic fraud data without any data leakage
    """
    
    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        
    def create_realistic_customers(self, n_customers=5000):
        """Create realistic customer profiles"""
        print(f"Creating {n_customers} realistic customer profiles...")
        
        customers = []
        
        for i in range(n_customers):
            # Realistic age distribution
            age = np.random.normal(40, 15)
            age = max(18, min(85, age))
            
            # Realistic tenure distribution (many new customers)
            tenure = np.random.exponential(scale=365) + 1
            tenure = min(tenure, 3650)  # Max 10 years
            
            # Income distribution
            income = np.random.lognormal(mean=10.5, sigma=0.6)
            income = max(20000, min(200000, income))
            
            # Spending habits (no fraud history)
            avg_amount = np.random.lognormal(mean=3.5, sigma=0.8)
            avg_amount = max(5, min(500, avg_amount))
            
            # Transaction frequency
            frequency = np.random.poisson(15) + 5
            
            # Device usage patterns
            mobile_preference = np.random.beta(2, 2)  # 0-1, higher = prefers mobile
            
            # Geographic patterns
            home_location_variety = np.random.exponential(scale=10) + 1
            
            customers.append({
                'customer_id': i + 1,
                'age': int(age),
                'tenure_days': int(tenure),
                'income': income,
                'avg_amount': avg_amount,
                'transaction_frequency': frequency,
                'mobile_preference': mobile_preference,
                'home_location_variety': home_location_variety,
                'preferred_merchant': np.random.choice(['retail', 'online', 'food', 'gas', 'travel']),
                'typical_hour_range': np.random.choice(['morning', 'afternoon', 'evening', 'night'])
            })
        
        return pd.DataFrame(customers)
    
    def generate_realistic_transactions(self, customers, n_transactions=100000, fraud_rate=0.02):
        """Generate realistic transactions with subtle fraud patterns"""
        print(f"Generating {n_transactions} transactions with {fraud_rate:.1%} fraud rate...")
        
        transactions = []
        fraud_count = 0
        legit_count = 0
        
        # Create merchant categories with realistic fraud rates
        merchant_fraud_rates = {
            'retail': 0.01,
            'online': 0.04,
            'food': 0.005,
            'gas': 0.008,
            'travel': 0.03,
            'electronics': 0.06,
            'healthcare': 0.002
        }
        
        for i in range(n_transactions):
            # Select customer
            customer = customers.iloc[np.random.randint(0, len(customers))]
            
            # Determine if this should be fraud (based on merchant and randomness)
            merchant_probs = list(merchant_fraud_rates.values())
            merchant_probs = [p/sum(merchant_probs) for p in merchant_probs]
            merchant = np.random.choice(list(merchant_fraud_rates.keys()), p=merchant_probs)
            base_fraud_prob = merchant_fraud_rates[merchant]
            
            # Adjust fraud probability based on customer characteristics
            if customer['tenure_days'] < 30:  # New customers
                base_fraud_prob *= 2.0
            if customer['income'] < 40000:  # Lower income
                base_fraud_prob *= 1.5
            
            is_fraud = np.random.random() < base_fraud_prob
            
            # Generate transaction time
            base_time = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            # Time patterns
            if customer['typical_hour_range'] == 'morning':
                hour = np.random.choice([6, 7, 8, 9, 10, 11])
            elif customer['typical_hour_range'] == 'afternoon':
                hour = np.random.choice([12, 13, 14, 15, 16, 17])
            elif customer['typical_hour_range'] == 'evening':
                hour = np.random.choice([18, 19, 20, 21])
            else:  # night
                hour = np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            
            transaction_time = base_time.replace(hour=hour)
            
            # Amount patterns
            if is_fraud:
                # Fraud transactions have different amount patterns
                if merchant == 'electronics':
                    amount = np.random.lognormal(mean=6.5, sigma=1.0)  # Higher amounts
                elif merchant == 'online':
                    amount = np.random.lognormal(mean=4.5, sigma=1.2)
                else:
                    amount = np.random.lognormal(mean=4.0, sigma=1.0)
            else:
                # Normal transactions follow customer's typical patterns
                amount = np.random.lognormal(mean=np.log(customer['avg_amount']), sigma=0.5)
            
            amount = max(1, min(10000, amount))
            
            # Location patterns
            if is_fraud:
                # Fraud often occurs far from usual locations
                distance_home = np.random.exponential(scale=50) + 10
                distance_last = np.random.exponential(scale=25) + 5
            else:
                # Normal transactions closer to home
                distance_home = np.random.exponential(scale=customer['home_location_variety'])
                distance_last = np.random.exponential(scale=5)
            
            # Device patterns
            if is_fraud:
                # Fraud often uses different devices
                devices_today = np.random.poisson(3) + 2
                is_mobile = np.random.random() < 0.7
            else:
                # Normal customer device patterns
                devices_today = max(1, np.random.poisson(2))
                is_mobile = np.random.random() < customer['mobile_preference']
            
            # Calculate ratio to customer's average (no leakage - using historical avg)
            ratio_to_median = amount / customer['avg_amount']
            
            # Create transaction record
            transaction = {
                'transaction_id': i + 1,
                'transaction_time': transaction_time,
                'transaction_amount': amount,
                'customer_id': customer['customer_id'],
                'customer_age': customer['age'],
                'customer_tenure_days': customer['tenure_days'],
                'merchant_category': merchant,
                'transaction_hour': hour,
                'distance_from_home_km': distance_home,
                'distance_from_last_transaction_km': distance_last,
                'devices_used_today': devices_today,
                'is_mobile_transaction': is_mobile,
                'ratio_to_median_purchase_price': ratio_to_median,
                'customer_avg_amount': customer['avg_amount'],
                'customer_income': customer['income'],
                'customer_mobile_preference': customer['mobile_preference'],
                'customer_home_location_variety': customer['home_location_variety'],
                'is_fraud': 1 if is_fraud else 0
            }
            
            transactions.append(transaction)
            
            if is_fraud:
                fraud_count += 1
            else:
                legit_count += 1
        
        df = pd.DataFrame(transactions)
        
        print(f"Generated: {len(df)} transactions")
        print(f"Fraud: {fraud_count} ({fraud_count/len(df):.3%})")
        print(f"Legitimate: {legit_count} ({legit_count/len(df):.3%})")
        
        return df
    
    def add_realistic_features(self, df):
        """Add realistic features without data leakage"""
        print("Adding realistic features...")
        
        # Time-based features
        df['transaction_day_of_week'] = df['transaction_time'].dt.dayofweek
        df['transaction_month'] = df['transaction_time'].dt.month
        df['is_weekend'] = df['transaction_day_of_week'] >= 5
        df['is_night_time'] = df['transaction_hour'].between(22, 6)
        df['is_business_hours'] = df['transaction_hour'].between(9, 17)
        
        # Amount-based features
        df['log_transaction_amount'] = np.log1p(df['transaction_amount'])
        df['is_high_amount'] = (df['transaction_amount'] > 500).astype(int)
        df['is_very_high_amount'] = (df['transaction_amount'] > 2000).astype(int)
        
        # Distance features
        df['total_distance_km'] = df['distance_from_home_km'] + df['distance_from_last_transaction_km']
        df['distance_ratio'] = df['distance_from_last_transaction_km'] / (df['distance_from_home_km'] + 1e-6)
        df['is_far_from_home'] = (df['distance_from_home_km'] > 30).astype(int)
        
        # Device features
        df['is_multiple_devices'] = (df['devices_used_today'] > 2).astype(int)
        
        # Customer behavior features (no leakage - only historical info)
        df['is_new_customer'] = (df['customer_tenure_days'] < 30).astype(int)
        df['is_young_customer'] = (df['customer_age'] < 25).astype(int)
        df['is_low_income'] = (df['customer_income'] < 40000).astype(int)
        
        # Spending pattern features
        df['is_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 3).astype(int)
        df['is_very_unusual_spending'] = (df['ratio_to_median_purchase_price'] > 5).astype(int)
        
        # Time risk categories
        df['time_risk_category'] = pd.cut(df['transaction_hour'],
                                        bins=[0, 6, 12, 18, 24],
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                        ordered=False)
        
        # Amount categories
        df['amount_category'] = pd.cut(df['transaction_amount'],
                                      bins=[0, 25, 100, 500, 2000, float('inf')],
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        return df
    
    def create_realistic_dataset(self, n_customers=5000, n_transactions=100000, fraud_rate=0.02):
        """Create complete realistic dataset"""
        print("="*60)
        print("CREATING REALISTIC DATASET (NO DATA LEAKAGE)")
        print("="*60)
        
        # Create customers
        customers = self.create_realistic_customers(n_customers)
        
        # Generate transactions
        df = self.generate_realistic_transactions(customers, n_transactions, fraud_rate)
        
        # Add realistic features
        df = self.add_realistic_features(df)
        
        # Select final columns (no leakage)
        final_columns = [
            'transaction_amount', 'customer_age', 'customer_tenure_days',
            'merchant_category', 'transaction_hour', 'is_weekend', 'is_night_time', 'is_business_hours',
            'distance_from_home_km', 'distance_from_last_transaction_km',
            'devices_used_today', 'is_mobile_transaction',
            'ratio_to_median_purchase_price',
            'customer_avg_amount', 'customer_income', 'customer_mobile_preference', 'customer_home_location_variety',
            'transaction_day_of_week', 'transaction_month',
            'log_transaction_amount', 'is_high_amount', 'is_very_high_amount',
            'total_distance_km', 'distance_ratio', 'is_far_from_home',
            'is_multiple_devices', 'is_new_customer', 'is_young_customer', 'is_low_income',
            'is_unusual_spending', 'is_very_unusual_spending',
            'time_risk_category', 'amount_category',
            'is_fraud'
        ]
        
        df = df[final_columns]
        
        # Create model-ready version
        df_model = df.copy()
        
        # Handle categorical columns
        categorical_columns = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            dummies = pd.get_dummies(df_model[col], prefix=col, drop_first=True)
            df_model = pd.concat([df_model, dummies], axis=1)
            df_model.drop(col, axis=1, inplace=True)
        
        print(f"\nRealistic dataset created!")
        print(f"Total transactions: {len(df)}")
        print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
        print(f"Features: {len(df_model.columns)}")
        
        return df, df_model
    
    def save_dataset(self, df, df_model, filepath='data/realistic_fraud_data.csv'):
        """Save realistic dataset"""
        df.to_csv(filepath, index=False)
        print(f"Realistic dataset saved to {filepath}")
        
        model_filepath = filepath.replace('.csv', '_model_ready.csv')
        df_model.to_csv(model_filepath, index=False)
        print(f"Model-ready dataset saved to {model_filepath}")

def main():
    """Main function"""
    generator = RealisticDataGenerator()
    
    # Create realistic dataset with 2% fraud rate (realistic)
    df, df_model = generator.create_realistic_dataset(
        n_customers=5000, 
        n_transactions=100000, 
        fraud_rate=0.02
    )
    
    # Save datasets
    generator.save_dataset(df, df_model)
    
    # Display statistics
    print(f"\n{'='*60}")
    print("REALISTIC DATASET STATISTICS")
    print("="*60)
    
    print(f"Fraud by Merchant Category:")
    fraud_by_merchant = df[df['is_fraud'] == 1]['merchant_category'].value_counts()
    for merchant, count in fraud_by_merchant.items():
        merchant_total = df[df['merchant_category'] == merchant].shape[0]
        fraud_rate = count / merchant_total
        print(f"  {merchant}: {count} ({fraud_rate:.1%})")
    
    print(f"\nKey Fraud Indicators in Fraudulent Transactions:")
    print(f"  Unusual spending (>3x): {df[df['is_fraud'] == 1]['is_unusual_spending'].mean():.1%}")
    print(f"  Multiple devices (>2): {df[df['is_fraud'] == 1]['is_multiple_devices'].mean():.1%}")
    print(f"  Far from home (>30km): {df[df['is_fraud'] == 1]['is_far_from_home'].mean():.1%}")
    print(f"  Night time: {df[df['is_fraud'] == 1]['is_night_time'].mean():.1%}")
    print(f"  New customers: {df[df['is_fraud'] == 1]['is_new_customer'].mean():.1%}")
    
    return df, df_model

if __name__ == "__main__":
    df, df_model = main()
