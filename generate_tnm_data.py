"""
Generate TNM broadband customer dataset for analysis
This creates a realistic dataset based on TNM operations in Malawi
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_tnm_broadband_data(n_customers=10000001):
    """
    Generate TNM broadband customer dataset
    Note: Generating 10+ million records may take several minutes
    Uses vectorized operations for efficiency
    """
    np.random.seed(42)
    
    print(f"Generating {n_customers:,} customer records...")
    
    # TNM regions in Malawi
    regions = ['Northern', 'Central', 'Southern', 'Lilongwe', 'Blantyre']
    region_weights = [0.15, 0.20, 0.30, 0.20, 0.15]  # Southern region is larger
    
    # TNM subscription plans (Malawi context)
    subscription_plans = ['Basic', 'Standard', 'Premium', 'Unlimited']
    plan_weights = [0.30, 0.35, 0.20, 0.15]
    plan_fees = {
        'Basic': 15000,      # MWK per month
        'Standard': 25000,
        'Premium': 40000,
        'Unlimited': 60000
    }
    
    # Generate customer IDs (optimized for large datasets)
    print("Generating customer IDs...")
    # Use more efficient string formatting for large datasets
    max_digits = len(str(n_customers))
    customer_ids = [f'TNM{str(i+1).zfill(max_digits)}' for i in range(n_customers)]
    
    # Generate regions (vectorized)
    print("Generating regions...")
    regions_array = np.random.choice(regions, n_customers, p=region_weights)
    
    # Generate subscription plans (vectorized)
    print("Generating subscription plans...")
    plans_array = np.random.choice(subscription_plans, n_customers, p=plan_weights)
    
    # Generate signup dates (vectorized)
    print("Generating signup dates...")
    days_ago = np.random.randint(30, 730, n_customers)
    base_date = datetime.now()
    signup_dates = [base_date - timedelta(days=int(d)) for d in days_ago]
    
    # Generate monthly fees based on plan (vectorized)
    print("Generating monthly fees...")
    fee_dict = {plan: plan_fees[plan] for plan in subscription_plans}
    monthly_fees = np.array([fee_dict[plan] for plan in plans_array])
    
    # Generate data usage (GB) - varies by plan (vectorized)
    print("Generating data usage...")
    usage_by_plan = {
        'Basic': (20, 8),
        'Standard': (50, 15),
        'Premium': (100, 25),
        'Unlimited': (150, 40)
    }
    
    data_usage = np.zeros(n_customers)
    for plan in subscription_plans:
        mask = plans_array == plan
        mean, std = usage_by_plan[plan]
        data_usage[mask] = np.maximum(0, np.random.normal(mean, std, mask.sum()))
    
    # Generate network downtime (hours per month) - higher in Southern region (vectorized)
    print("Generating network downtime...")
    southern_mask = regions_array == 'Southern'
    downtime = np.zeros(n_customers)
    downtime[southern_mask] = np.minimum(50, np.random.exponential(8, southern_mask.sum()))
    downtime[~southern_mask] = np.minimum(50, np.random.exponential(4, (~southern_mask).sum()))
    
    # Generate churn - influenced by downtime, usage, and region (vectorized)
    print("Generating churn status...")
    churn_prob = np.full(n_customers, 0.1)  # Base probability
    churn_prob += np.minimum(0.4, downtime / 50)  # Higher downtime = higher churn
    churn_prob -= np.minimum(0.2, data_usage / 200)  # Higher usage = lower churn
    churn_prob[southern_mask] += 0.15  # Southern region has higher churn
    churn_prob = np.clip(churn_prob, 0.05, 0.7)
    
    churned = np.where(np.random.random(n_customers) < churn_prob, 'Yes', 'No')
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'region': regions_array,
        'subscription_plan': plans_array,
        'signup_date': signup_dates,
        'monthly_fee': monthly_fees,
        'data_usage_gb': data_usage,
        'network_downtime_hours': downtime,
        'churned': churned
    })
    
    # Save to CSV
    print("Saving to CSV (this may take a few minutes for large datasets)...")
    df.to_csv('tnm_broadband.csv', index=False)
    print(f"[OK] Generated TNM broadband dataset with {n_customers:,} customers")
    print(f"[OK] Saved to 'tnm_broadband.csv'")
    
    return df

if __name__ == "__main__":
    print("Generating 10,000,001 customer records (this may take several minutes)...")
    df = generate_tnm_broadband_data(10000001)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:")
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
