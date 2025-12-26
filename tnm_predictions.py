"""
TNM Predictive Analysis: What-if Scenarios & 5-Year Forecast
1. Predicts outcomes if TNM fixes identified problems
2. Forecasts business metrics for next 5 years
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Create prediction folder if it doesn't exist
prediction_folder = 'prediction'
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)
    print(f"[OK] Created '{prediction_folder}' folder")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 70)
print("TNM PREDICTIVE ANALYSIS: IMPROVEMENT SCENARIOS & 5-YEAR FORECAST")
print("=" * 70)

# ============================================================================
# Load Current Data
# ============================================================================
print("\n[STEP 1] Loading current TNM data...")
df = pd.read_csv("tnm_broadband.csv")
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['monthly_revenue'] = df['monthly_fee']

print(f"[OK] Loaded {len(df):,} customer records")

# For large datasets, sample for prediction calculations to speed up processing
# We'll use statistical sampling for scenarios but full data for base metrics
sample_size = min(50000, len(df))  # Use 50k sample or full dataset if smaller
if len(df) > sample_size:
    print(f"[INFO] Using {sample_size:,} sample records for scenario calculations to optimize performance")
    df_sample = df.sample(n=sample_size, random_state=42)
else:
    df_sample = df.copy()

# Calculate current metrics
current_total_customers = len(df)
current_churned = len(df[df['churned'] == 'Yes'])
current_churn_rate = (current_churned / current_total_customers) * 100
current_revenue = df['monthly_revenue'].sum()
current_avg_downtime = df['network_downtime_hours'].mean()
current_southern_downtime = df[df['region'] == 'Southern']['network_downtime_hours'].mean()
current_other_downtime = df[df['region'] != 'Southern']['network_downtime_hours'].mean()

print(f"\nCurrent State:")
print(f"  - Total Customers: {current_total_customers:,}")
print(f"  - Churned Customers: {current_churned:,}")
print(f"  - Current Churn Rate: {current_churn_rate:.2f}%")
print(f"  - Monthly Revenue: MWK {current_revenue:,.0f}")
print(f"  - Average Downtime: {current_avg_downtime:.2f} hours")
print(f"  - Southern Region Downtime: {current_southern_downtime:.2f} hours")
print(f"  - Other Regions Downtime: {current_other_downtime:.2f} hours")

# ============================================================================
# PART 1: WHAT-IF SCENARIOS - Impact of Fixing Problems
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: WHAT-IF SCENARIOS - Impact of Fixing Problems")
print("=" * 70)

def calculate_churn_probability(downtime, usage, region, base_prob=0.1):
    """Calculate churn probability based on factors"""
    churn_prob = base_prob
    churn_prob += min(0.4, downtime / 50)  # Higher downtime = higher churn
    churn_prob -= min(0.2, usage / 200)  # Higher usage = lower churn
    if region == 'Southern':
        churn_prob += 0.15  # Southern region penalty
    churn_prob = max(0.05, min(0.7, churn_prob))
    return churn_prob

# Scenario 1: Fix Network Downtime in Southern Region
print("\n[SCENARIO 1] Fixing Network Downtime in Southern Region")
print("-" * 70)

# Simulate improved downtime (reduce to match other regions)
df_scenario1 = df_sample.copy()
southern_mask = df_scenario1['region'] == 'Southern'
improved_downtime = df_scenario1.loc[southern_mask, 'network_downtime_hours'] * 0.5  # Reduce by 50%
df_scenario1.loc[southern_mask, 'network_downtime_hours'] = improved_downtime

# Recalculate churn probabilities (vectorized for efficiency)
churn_probs_scenario1 = np.array([
    calculate_churn_probability(
        row['network_downtime_hours'],
        row['data_usage_gb'],
        row['region']
    ) for _, row in df_scenario1.iterrows()
])

# Estimate new churn (using probabilities) - scale to full dataset
estimated_churn_scenario1 = (churn_probs_scenario1.mean() * len(df)) * (len(df) / len(df_scenario1))
new_churn_rate_scenario1 = (churn_probs_scenario1.mean()) * 100
churn_reduction_scenario1 = current_churn_rate - new_churn_rate_scenario1
customers_saved_scenario1 = (current_churn_rate - new_churn_rate_scenario1) / 100 * current_total_customers

# Revenue impact (assume saved customers continue paying)
avg_monthly_fee = df['monthly_fee'].mean()
revenue_increase_scenario1 = customers_saved_scenario1 * avg_monthly_fee

print(f"\nResults:")
print(f"  - New Average Downtime (Southern): {df_scenario1[southern_mask]['network_downtime_hours'].mean():.2f} hours")
print(f"  - Estimated New Churn Rate: {new_churn_rate_scenario1:.2f}%")
print(f"  - Churn Rate Reduction: {churn_reduction_scenario1:.2f} percentage points")
print(f"  - Customers Retained: ~{customers_saved_scenario1:,.0f}")
print(f"  - Monthly Revenue Increase: MWK {revenue_increase_scenario1:,.0f}")
print(f"  - Annual Revenue Increase: MWK {revenue_increase_scenario1 * 12:,.0f}")

# Scenario 2: Fix All Network Downtime (All Regions)
print("\n[SCENARIO 2] Fixing Network Downtime Across All Regions")
print("-" * 70)

df_scenario2 = df_sample.copy()
# Reduce downtime by 60% across all regions
df_scenario2['network_downtime_hours'] = df_scenario2['network_downtime_hours'] * 0.4

churn_probs_scenario2 = np.array([
    calculate_churn_probability(
        row['network_downtime_hours'],
        row['data_usage_gb'],
        row['region']
    ) for _, row in df_scenario2.iterrows()
])

new_churn_rate_scenario2 = (churn_probs_scenario2.mean()) * 100
churn_reduction_scenario2 = current_churn_rate - new_churn_rate_scenario2
customers_saved_scenario2 = (current_churn_rate - new_churn_rate_scenario2) / 100 * current_total_customers
revenue_increase_scenario2 = customers_saved_scenario2 * avg_monthly_fee

print(f"\nResults:")
print(f"  - New Average Downtime (All Regions): {df_scenario2['network_downtime_hours'].mean():.2f} hours")
print(f"  - Estimated New Churn Rate: {new_churn_rate_scenario2:.2f}%")
print(f"  - Churn Rate Reduction: {churn_reduction_scenario2:.2f} percentage points")
print(f"  - Customers Retained: ~{customers_saved_scenario2:,.0f}")
print(f"  - Monthly Revenue Increase: MWK {revenue_increase_scenario2:,.0f}")
print(f"  - Annual Revenue Increase: MWK {revenue_increase_scenario2 * 12:,.0f}")

# Scenario 3: Upgrade Customers from Basic/Standard to Premium/Unlimited
print("\n[SCENARIO 3] Upgrading Basic & Standard Customers")
print("-" * 70)

df_scenario3 = df_sample.copy()
upgrade_mask = df_scenario3['subscription_plan'].isin(['Basic', 'Standard'])

# Increase usage for upgraded customers (they use more with better plans)
df_scenario3.loc[upgrade_mask, 'data_usage_gb'] = df_scenario3.loc[upgrade_mask, 'data_usage_gb'] * 1.5

# Increase revenue (upgrade fees)
plan_upgrades = {
    'Basic': 25000,  # Upgrade to Standard fee
    'Standard': 40000  # Upgrade to Premium fee
}
for plan in ['Basic', 'Standard']:
    mask = (df_scenario3['subscription_plan'] == plan)
    df_scenario3.loc[mask, 'monthly_fee'] = plan_upgrades[plan]
    df_scenario3.loc[mask, 'monthly_revenue'] = plan_upgrades[plan]

churn_probs_scenario3 = np.array([
    calculate_churn_probability(
        row['network_downtime_hours'],
        row['data_usage_gb'],
        row['region']
    ) for _, row in df_scenario3.iterrows()
])

new_churn_rate_scenario3 = (churn_probs_scenario3.mean()) * 100
churn_reduction_scenario3 = current_churn_rate - new_churn_rate_scenario3
customers_saved_scenario3 = (current_churn_rate - new_churn_rate_scenario3) / 100 * current_total_customers

# Calculate revenue increase from upgrades (scale to full dataset)
upgrade_mask_full = df['subscription_plan'].isin(['Basic', 'Standard'])
revenue_increase_from_upgrades = 0
for plan in ['Basic', 'Standard']:
    mask = (df['subscription_plan'] == plan)
    count = mask.sum()
    old_revenue = count * df[df['subscription_plan'] == plan]['monthly_fee'].iloc[0]
    new_revenue = count * plan_upgrades[plan]
    revenue_increase_from_upgrades += (new_revenue - old_revenue)
revenue_increase_scenario3 = revenue_increase_from_upgrades + (customers_saved_scenario3 * avg_monthly_fee)

print(f"\nResults:")
print(f"  - Customers Upgraded: {upgrade_mask.sum():,}")
print(f"  - Estimated New Churn Rate: {new_churn_rate_scenario3:.2f}%")
print(f"  - Churn Rate Reduction: {churn_reduction_scenario3:.2f} percentage points")
print(f"  - Customers Retained: ~{customers_saved_scenario3:,.0f}")
print(f"  - Monthly Revenue Increase: MWK {revenue_increase_scenario3:,.0f}")
print(f"  - Annual Revenue Increase: MWK {revenue_increase_scenario3 * 12:,.0f}")

# Scenario 4: Combined Improvements (All fixes together)
print("\n[SCENARIO 4] Combined Improvements (All Fixes)")
print("-" * 70)

df_scenario4 = df_sample.copy()
# Fix downtime
southern_mask = df_scenario4['region'] == 'Southern'
df_scenario4.loc[southern_mask, 'network_downtime_hours'] *= 0.5
df_scenario4.loc[~southern_mask, 'network_downtime_hours'] *= 0.4

# Upgrade customers
upgrade_mask = df_scenario4['subscription_plan'].isin(['Basic', 'Standard'])
df_scenario4.loc[upgrade_mask, 'data_usage_gb'] *= 1.5
for plan in ['Basic', 'Standard']:
    mask = (df_scenario4['subscription_plan'] == plan)
    df_scenario4.loc[mask, 'monthly_fee'] = plan_upgrades[plan]
    df_scenario4.loc[mask, 'monthly_revenue'] = plan_upgrades[plan]

churn_probs_scenario4 = np.array([
    calculate_churn_probability(
        row['network_downtime_hours'],
        row['data_usage_gb'],
        row['region']
    ) for _, row in df_scenario4.iterrows()
])

new_churn_rate_scenario4 = (churn_probs_scenario4.mean()) * 100
churn_reduction_scenario4 = current_churn_rate - new_churn_rate_scenario4
customers_saved_scenario4 = (current_churn_rate - new_churn_rate_scenario4) / 100 * current_total_customers

# Combined revenue increase (upgrades + retained customers)
revenue_increase_scenario4 = revenue_increase_from_upgrades + (customers_saved_scenario4 * avg_monthly_fee)

print(f"\nResults:")
print(f"  - New Average Downtime: {df_scenario4['network_downtime_hours'].mean():.2f} hours")
print(f"  - Estimated New Churn Rate: {new_churn_rate_scenario4:.2f}%")
print(f"  - Churn Rate Reduction: {churn_reduction_scenario4:.2f} percentage points")
print(f"  - Customers Retained: ~{customers_saved_scenario4:,.0f}")
print(f"  - Monthly Revenue Increase: MWK {revenue_increase_scenario4:,.0f}")
print(f"  - Annual Revenue Increase: MWK {revenue_increase_scenario4 * 12:,.0f}")

# Visualization: Scenario Comparison
print("\n[PLOT] Creating scenario comparison visualization...")
scenarios = ['Current', 'Fix Southern\nDowntime', 'Fix All\nDowntime', 'Customer\nUpgrades', 'Combined\nImprovements']
churn_rates = [current_churn_rate, new_churn_rate_scenario1, new_churn_rate_scenario2, 
               new_churn_rate_scenario3, new_churn_rate_scenario4]
revenue_increases = [0, revenue_increase_scenario1, revenue_increase_scenario2, 
                     revenue_increase_scenario3, revenue_increase_scenario4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Churn Rate Comparison
ax1.bar(scenarios, churn_rates, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
ax1.set_title('Churn Rate by Scenario - TNM', fontsize=14, fontweight='bold')
ax1.set_ylabel('Churn Rate (%)', fontsize=12)
ax1.set_ylim(0, max(churn_rates) * 1.2)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(churn_rates):
    ax1.text(i, v + max(churn_rates) * 0.02, f'{v:.2f}%', ha='center', fontweight='bold')

# Revenue Increase Comparison
ax2.bar(scenarios, [r/1e9 for r in revenue_increases], color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
ax2.set_title('Monthly Revenue Increase by Scenario - TNM', fontsize=14, fontweight='bold')
ax2.set_ylabel('Revenue Increase (Billion MWK)', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(revenue_increases):
    if v > 0:
        ax2.text(i, v/1e9 + max(revenue_increases)/1e9 * 0.02, 
                f'MWK {v/1e9:.2f}B', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
scenario_comparison_path = os.path.join(prediction_folder, 'tnm_scenario_comparison.png')
plt.savefig(scenario_comparison_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved plot: {scenario_comparison_path}")
plt.show()

# ============================================================================
# PART 2: 5-YEAR FORECAST
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: 5-YEAR BUSINESS FORECAST")
print("=" * 70)

# Base metrics for forecasting
base_customers = current_total_customers - current_churned  # Active customers
base_monthly_revenue = current_revenue
base_churn_rate = current_churn_rate / 100  # Convert to decimal
monthly_growth_rate = 0.02  # 2% monthly customer growth (conservative)
annual_growth_rate = 0.15  # 15% annual market growth

# Forecast parameters
years = 5
months = years * 12

# Initialize forecast arrays
forecast_months = []
forecast_customers = []
forecast_revenue = []
forecast_churn_customers = []
forecast_churn_rate = []

# Scenario A: No Improvements (Current State)
print("\n[FORECAST A] 5-Year Forecast: No Improvements (Current State)")
print("-" * 70)

current_churn = base_churn_rate
current_customers = base_customers
current_rev = base_monthly_revenue

for month in range(1, months + 1):
    # Customer growth (with churn)
    new_customers = current_customers * monthly_growth_rate
    churned = current_customers * current_churn
    current_customers = current_customers + new_customers - churned
    
    # Revenue (affected by churn and growth)
    current_rev = current_customers * avg_monthly_fee
    
    if month % 12 == 0:
        year = month // 12
        forecast_months.append(f"Year {year}")
        forecast_customers.append(current_customers)
        forecast_revenue.append(current_rev)
        forecast_churn_customers.append(churned * 12)  # Annual churn
        forecast_churn_rate.append(current_churn * 100)

year1_customers_no_improve = forecast_customers[0]
year5_customers_no_improve = forecast_customers[-1]
year1_revenue_no_improve = forecast_revenue[0]
year5_revenue_no_improve = forecast_revenue[-1]

print(f"\nForecast Summary (No Improvements):")
print(f"  Year 1 - Customers: {year1_customers_no_improve:,.0f}, Revenue: MWK {year1_revenue_no_improve:,.0f}")
print(f"  Year 5 - Customers: {year5_customers_no_improve:,.0f}, Revenue: MWK {year5_revenue_no_improve:,.0f}")
print(f"  5-Year Customer Growth: {((year5_customers_no_improve / base_customers) - 1) * 100:.1f}%")
print(f"  5-Year Revenue Growth: {((year5_revenue_no_improve / base_monthly_revenue) - 1) * 100:.1f}%")

# Scenario B: With Improvements (Combined Improvements)
print("\n[FORECAST B] 5-Year Forecast: With Improvements")
print("-" * 70)

# Reset for improved scenario
improved_churn = new_churn_rate_scenario4 / 100  # Use combined improvements churn rate
improved_customers = base_customers + customers_saved_scenario4
improved_rev = base_monthly_revenue + revenue_increase_scenario4
improved_avg_fee = improved_rev / improved_customers  # Slightly higher due to upgrades

forecast_customers_improved = []
forecast_revenue_improved = []
forecast_churn_customers_improved = []

current_customers = improved_customers
current_rev = improved_rev
current_churn = improved_churn

for month in range(1, months + 1):
    # Customer growth (with reduced churn)
    new_customers = current_customers * monthly_growth_rate
    churned = current_customers * current_churn
    current_customers = current_customers + new_customers - churned
    
    # Revenue (slightly higher ARPU due to plan upgrades)
    current_rev = current_customers * improved_avg_fee
    
    if month % 12 == 0:
        year = month // 12
        forecast_customers_improved.append(current_customers)
        forecast_revenue_improved.append(current_rev)
        forecast_churn_customers_improved.append(churned * 12)

year1_customers_improved = forecast_customers_improved[0]
year5_customers_improved = forecast_customers_improved[-1]
year1_revenue_improved = forecast_revenue_improved[0]
year5_revenue_improved = forecast_revenue_improved[-1]

print(f"\nForecast Summary (With Improvements):")
print(f"  Year 1 - Customers: {year1_customers_improved:,.0f}, Revenue: MWK {year1_revenue_improved:,.0f}")
print(f"  Year 5 - Customers: {year5_customers_improved:,.0f}, Revenue: MWK {year5_revenue_improved:,.0f}")
print(f"  5-Year Customer Growth: {((year5_customers_improved / improved_customers) - 1) * 100:.1f}%")
print(f"  5-Year Revenue Growth: {((year5_revenue_improved / improved_rev) - 1) * 100:.1f}%")

# Calculate the difference
customer_gain = year5_customers_improved - year5_customers_no_improve
revenue_gain = year5_revenue_improved - year5_revenue_no_improve

print(f"\nImpact of Improvements by Year 5:")
print(f"  Additional Customers: +{customer_gain:,.0f}")
print(f"  Additional Revenue: MWK {revenue_gain:,.0f} per month")
print(f"  Additional Annual Revenue: MWK {revenue_gain * 12:,.0f}")

# Visualization: 5-Year Forecast
print("\n[PLOT] Creating 5-year forecast visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

years_list = [1, 2, 3, 4, 5]

# Customer Forecast
ax1.plot(years_list, forecast_customers, marker='o', linewidth=2, label='No Improvements', color='#d62728')
ax1.plot(years_list, forecast_customers_improved, marker='s', linewidth=2, label='With Improvements', color='#2ca02c')
ax1.set_title('5-Year Customer Forecast - TNM', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Customers', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(years_list)

# Revenue Forecast
ax2.plot(years_list, [r/1e9 for r in forecast_revenue], marker='o', linewidth=2, 
         label='No Improvements', color='#d62728')
ax2.plot(years_list, [r/1e9 for r in forecast_revenue_improved], marker='s', linewidth=2, 
         label='With Improvements', color='#2ca02c')
ax2.set_title('5-Year Monthly Revenue Forecast - TNM', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Monthly Revenue (Billion MWK)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(years_list)

plt.tight_layout()
forecast_path = os.path.join(prediction_folder, 'tnm_5year_forecast.png')
plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved plot: {forecast_path}")
plt.show()

# ============================================================================
# Save Detailed Predictions to File
# ============================================================================
print("\n" + "=" * 70)
print("SAVING DETAILED PREDICTIONS")
print("=" * 70)

# Create comprehensive predictions dictionary
predictions_data = {
    'current_state': {
        'total_customers': int(current_total_customers),
        'churned_customers': int(current_churned),
        'churn_rate_percent': float(current_churn_rate),
        'monthly_revenue_mwk': float(current_revenue),
        'average_downtime_hours': float(current_avg_downtime),
        'southern_downtime_hours': float(current_southern_downtime),
        'other_regions_downtime_hours': float(current_other_downtime)
    },
    'scenario_1_fix_southern_downtime': {
        'churn_rate_percent': float(new_churn_rate_scenario1),
        'churn_reduction_percentage_points': float(churn_reduction_scenario1),
        'customers_retained': float(customers_saved_scenario1),
        'monthly_revenue_increase_mwk': float(revenue_increase_scenario1),
        'annual_revenue_increase_mwk': float(revenue_increase_scenario1 * 12)
    },
    'scenario_2_fix_all_downtime': {
        'churn_rate_percent': float(new_churn_rate_scenario2),
        'churn_reduction_percentage_points': float(churn_reduction_scenario2),
        'customers_retained': float(customers_saved_scenario2),
        'monthly_revenue_increase_mwk': float(revenue_increase_scenario2),
        'annual_revenue_increase_mwk': float(revenue_increase_scenario2 * 12)
    },
    'scenario_3_customer_upgrades': {
        'churn_rate_percent': float(new_churn_rate_scenario3),
        'churn_reduction_percentage_points': float(churn_reduction_scenario3),
        'customers_retained': float(customers_saved_scenario3),
        'monthly_revenue_increase_mwk': float(revenue_increase_scenario3),
        'annual_revenue_increase_mwk': float(revenue_increase_scenario3 * 12)
    },
    'scenario_4_combined_improvements': {
        'churn_rate_percent': float(new_churn_rate_scenario4),
        'churn_reduction_percentage_points': float(churn_reduction_scenario4),
        'customers_retained': float(customers_saved_scenario4),
        'monthly_revenue_increase_mwk': float(revenue_increase_scenario4),
        'annual_revenue_increase_mwk': float(revenue_increase_scenario4 * 12)
    },
    'forecast_5year_no_improvements': {
        'year_1': {
            'customers': float(year1_customers_no_improve),
            'monthly_revenue_mwk': float(year1_revenue_no_improve)
        },
        'year_5': {
            'customers': float(year5_customers_no_improve),
            'monthly_revenue_mwk': float(year5_revenue_no_improve)
        },
        'customer_growth_percent': float(((year5_customers_no_improve / base_customers) - 1) * 100),
        'revenue_growth_percent': float(((year5_revenue_no_improve / base_monthly_revenue) - 1) * 100)
    },
    'forecast_5year_with_improvements': {
        'year_1': {
            'customers': float(year1_customers_improved),
            'monthly_revenue_mwk': float(year1_revenue_improved)
        },
        'year_5': {
            'customers': float(year5_customers_improved),
            'monthly_revenue_mwk': float(year5_revenue_improved)
        },
        'customer_growth_percent': float(((year5_customers_improved / improved_customers) - 1) * 100),
        'revenue_growth_percent': float(((year5_revenue_improved / improved_rev) - 1) * 100),
        'additional_customers_vs_no_improvements': float(customer_gain),
        'additional_monthly_revenue_vs_no_improvements_mwk': float(revenue_gain),
        'additional_annual_revenue_vs_no_improvements_mwk': float(revenue_gain * 12),
        'cumulative_5year_additional_revenue_mwk': float(revenue_gain * 12 * 5)
    },
    'detailed_forecast_by_year': {
        'no_improvements': {
            'customers_by_year': [float(x) for x in forecast_customers],
            'revenue_by_year_mwk': [float(x) for x in forecast_revenue]
        },
        'with_improvements': {
            'customers_by_year': [float(x) for x in forecast_customers_improved],
            'revenue_by_year_mwk': [float(x) for x in forecast_revenue_improved]
        }
    },
    'scenario_comparison': {
        'scenarios': ['Current', 'Fix Southern Downtime', 'Fix All Downtime', 'Customer Upgrades', 'Combined Improvements'],
        'churn_rates_percent': [float(x) for x in churn_rates],
        'revenue_increases_mwk': [float(x) for x in revenue_increases]
    }
}

# Save as JSON
json_path = os.path.join(prediction_folder, 'detailed_predictions.json')
with open(json_path, 'w') as f:
    json.dump(predictions_data, f, indent=2)
print(f"[OK] Saved detailed predictions JSON: {json_path}")

# Save as CSV for scenario comparison
scenario_df = pd.DataFrame({
    'Scenario': predictions_data['scenario_comparison']['scenarios'],
    'Churn_Rate_Percent': predictions_data['scenario_comparison']['churn_rates_percent'],
    'Monthly_Revenue_Increase_MWK': predictions_data['scenario_comparison']['revenue_increases_mwk'],
    'Annual_Revenue_Increase_MWK': [x * 12 for x in predictions_data['scenario_comparison']['revenue_increases_mwk']]
})
csv_path = os.path.join(prediction_folder, 'scenario_comparison.csv')
scenario_df.to_csv(csv_path, index=False)
print(f"[OK] Saved scenario comparison CSV: {csv_path}")

# Save 5-year forecast as CSV
forecast_df = pd.DataFrame({
    'Year': [1, 2, 3, 4, 5],
    'Customers_No_Improvements': predictions_data['detailed_forecast_by_year']['no_improvements']['customers_by_year'],
    'Revenue_No_Improvements_MWK': predictions_data['detailed_forecast_by_year']['no_improvements']['revenue_by_year_mwk'],
    'Customers_With_Improvements': predictions_data['detailed_forecast_by_year']['with_improvements']['customers_by_year'],
    'Revenue_With_Improvements_MWK': predictions_data['detailed_forecast_by_year']['with_improvements']['revenue_by_year_mwk'],
    'Additional_Customers': [y - x for x, y in zip(
        predictions_data['detailed_forecast_by_year']['no_improvements']['customers_by_year'],
        predictions_data['detailed_forecast_by_year']['with_improvements']['customers_by_year']
    )],
    'Additional_Revenue_MWK': [y - x for x, y in zip(
        predictions_data['detailed_forecast_by_year']['no_improvements']['revenue_by_year_mwk'],
        predictions_data['detailed_forecast_by_year']['with_improvements']['revenue_by_year_mwk']
    )]
})
forecast_csv_path = os.path.join(prediction_folder, '5year_forecast.csv')
forecast_df.to_csv(forecast_csv_path, index=False)
print(f"[OK] Saved 5-year forecast CSV: {forecast_csv_path}")

# Save summary text file
summary_path = os.path.join(prediction_folder, 'predictions_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("TNM PREDICTIVE ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("CURRENT STATE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Customers: {current_total_customers:,}\n")
    f.write(f"Churned Customers: {current_churned:,}\n")
    f.write(f"Current Churn Rate: {current_churn_rate:.2f}%\n")
    f.write(f"Monthly Revenue: MWK {current_revenue:,.0f}\n")
    f.write(f"Average Downtime: {current_avg_downtime:.2f} hours\n\n")
    
    f.write("SCENARIO PREDICTIONS\n")
    f.write("-" * 70 + "\n")
    f.write("\n1. Fix Network Downtime in Southern Region:\n")
    f.write(f"   - Churn Rate Reduction: {churn_reduction_scenario1:.2f} percentage points\n")
    f.write(f"   - Customers Retained: ~{customers_saved_scenario1:,.0f}\n")
    f.write(f"   - Monthly Revenue Increase: MWK {revenue_increase_scenario1:,.0f}\n")
    f.write(f"   - Annual Revenue Increase: MWK {revenue_increase_scenario1 * 12:,.0f}\n")
    
    f.write("\n2. Fix All Network Downtime:\n")
    f.write(f"   - Churn Rate Reduction: {churn_reduction_scenario2:.2f} percentage points\n")
    f.write(f"   - Customers Retained: ~{customers_saved_scenario2:,.0f}\n")
    f.write(f"   - Monthly Revenue Increase: MWK {revenue_increase_scenario2:,.0f}\n")
    f.write(f"   - Annual Revenue Increase: MWK {revenue_increase_scenario2 * 12:,.0f}\n")
    
    f.write("\n3. Customer Plan Upgrades:\n")
    f.write(f"   - Churn Rate Reduction: {churn_reduction_scenario3:.2f} percentage points\n")
    f.write(f"   - Customers Retained: ~{customers_saved_scenario3:,.0f}\n")
    f.write(f"   - Monthly Revenue Increase: MWK {revenue_increase_scenario3:,.0f}\n")
    f.write(f"   - Annual Revenue Increase: MWK {revenue_increase_scenario3 * 12:,.0f}\n")
    
    f.write("\n4. Combined Improvements (All Fixes):\n")
    f.write(f"   - Churn Rate Reduction: {churn_reduction_scenario4:.2f} percentage points\n")
    f.write(f"   - Customers Retained: ~{customers_saved_scenario4:,.0f}\n")
    f.write(f"   - Monthly Revenue Increase: MWK {revenue_increase_scenario4:,.0f}\n")
    f.write(f"   - Annual Revenue Increase: MWK {revenue_increase_scenario4 * 12:,.0f}\n")
    
    f.write("\n5-YEAR FORECAST\n")
    f.write("-" * 70 + "\n")
    f.write("\nWithout Improvements:\n")
    f.write(f"  Year 5 Customers: {year5_customers_no_improve:,.0f}\n")
    f.write(f"  Year 5 Monthly Revenue: MWK {year5_revenue_no_improve:,.0f}\n")
    f.write(f"  5-Year Revenue Growth: {((year5_revenue_no_improve / base_monthly_revenue) - 1) * 100:.1f}%\n")
    
    f.write("\nWith Improvements:\n")
    f.write(f"  Year 5 Customers: {year5_customers_improved:,.0f} (+{customer_gain:,.0f} vs no improvements)\n")
    f.write(f"  Year 5 Monthly Revenue: MWK {year5_revenue_improved:,.0f} (+MWK {revenue_gain:,.0f} vs no improvements)\n")
    f.write(f"  5-Year Revenue Growth: {((year5_revenue_improved / improved_rev) - 1) * 100:.1f}%\n")
    f.write(f"  5-Year Cumulative Additional Revenue: MWK {revenue_gain * 12 * 5:,.0f}\n")
    
    f.write("\nKEY RECOMMENDATIONS\n")
    f.write("-" * 70 + "\n")
    f.write("1. PRIORITY: Fix Network Downtime - Highest impact on customer retention\n")
    f.write("2. PRIORITY: Customer Plan Upgrades - Increases revenue per customer\n")
    f.write("3. STRATEGIC: Combined Improvement Program - Maximum business impact\n")
    f.write(f"   Expected 5-year cumulative benefit: MWK {revenue_gain * 12 * 5 / 1e9:.2f} billion\n")

print(f"[OK] Saved predictions summary text: {summary_path}")

# ============================================================================
# SUMMARY & KEY PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY & KEY PREDICTIONS")
print("=" * 70)

print("\n[PREDICTION 1] Impact of Fixing Network Downtime in Southern Region:")
print(f"  - Reduces churn rate by {churn_reduction_scenario1:.2f} percentage points")
print(f"  - Saves ~{customers_saved_scenario1:,.0f} customers from churning")
print(f"  - Increases monthly revenue by MWK {revenue_increase_scenario1:,.0f}")

print("\n[PREDICTION 2] Impact of Fixing All Network Downtime:")
print(f"  - Reduces churn rate by {churn_reduction_scenario2:.2f} percentage points")
print(f"  - Saves ~{customers_saved_scenario2:,.0f} customers from churning")
print(f"  - Increases monthly revenue by MWK {revenue_increase_scenario2:,.0f}")

print("\n[PREDICTION 3] Impact of Customer Plan Upgrades:")
print(f"  - Reduces churn rate by {churn_reduction_scenario3:.2f} percentage points")
print(f"  - Saves ~{customers_saved_scenario3:,.0f} customers from churning")
print(f"  - Increases monthly revenue by MWK {revenue_increase_scenario3:,.0f}")

print("\n[PREDICTION 4] Impact of Combined Improvements:")
print(f"  - Reduces churn rate by {churn_reduction_scenario4:.2f} percentage points")
print(f"  - Saves ~{customers_saved_scenario4:,.0f} customers from churning")
print(f"  - Increases monthly revenue by MWK {revenue_increase_scenario4:,.0f}")
print(f"  - Increases annual revenue by MWK {revenue_increase_scenario4 * 12:,.0f}")

print("\n[PREDICTION 5] 5-Year Forecast Without Improvements:")
print(f"  - Year 5 Customers: {year5_customers_no_improve:,.0f}")
print(f"  - Year 5 Monthly Revenue: MWK {year5_revenue_no_improve:,.0f}")
print(f"  - 5-Year Revenue Growth: {((year5_revenue_no_improve / base_monthly_revenue) - 1) * 100:.1f}%")

print("\n[PREDICTION 6] 5-Year Forecast With Improvements:")
print(f"  - Year 5 Customers: {year5_customers_improved:,.0f} (+{customer_gain:,.0f} vs no improvements)")
print(f"  - Year 5 Monthly Revenue: MWK {year5_revenue_improved:,.0f} (+MWK {revenue_gain:,.0f} vs no improvements)")
print(f"  - 5-Year Revenue Growth: {((year5_revenue_improved / improved_rev) - 1) * 100:.1f}%")
print(f"  - 5-Year Cumulative Additional Revenue: MWK {revenue_gain * 12 * 5:,.0f}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS BASED ON PREDICTIONS")
print("=" * 70)

print("\n1. PRIORITY: Fix Network Downtime")
print("   - Highest impact on customer retention")
print("   - Immediate revenue protection")
print("   - Estimated ROI: Very High")

print("\n2. PRIORITY: Customer Plan Upgrades")
print("   - Increases revenue per customer")
print("   - Reduces churn risk")
print("   - Estimated ROI: High")

print("\n3. STRATEGIC: Combined Improvement Program")
print("   - Maximum business impact")
print("   - Positions TNM for sustainable growth")
print("   - Expected 5-year cumulative benefit: MWK {:.0f} billion".format(revenue_gain * 12 * 5 / 1e9))

print("\n" + "=" * 70)
print("Predictive Analysis Complete!")
print("=" * 70)
print(f"\nAll outputs saved to '{prediction_folder}' folder:")
print(f"  - tnm_scenario_comparison.png")
print(f"  - tnm_5year_forecast.png")
print(f"  - detailed_predictions.json")
print(f"  - scenario_comparison.csv")
print(f"  - 5year_forecast.csv")
print(f"  - predictions_summary.txt")

