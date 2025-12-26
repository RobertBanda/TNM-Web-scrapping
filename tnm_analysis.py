"""
TNM Broadband Data Analysis
Following the 8-step analysis structure
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("TNM BROADBAND DATA ANALYSIS")
print("=" * 60)

# ============================================================================
# Step 1: Import Libraries & Data
# ============================================================================
print("\n" + "=" * 60)
print("Step 1: Import Libraries & Data")
print("=" * 60)

print("Loading dataset (this may take a moment for large files)...")
df = pd.read_csv("tnm_broadband.csv")
print(f"\n[OK] Loaded dataset with {len(df):,} customers")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# Step 2: Understand the Data
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: Understand the Data")
print("=" * 60)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nBusiness Understanding:")
print("• Customers by region")
print("• Subscription plans")
print("• Usage behavior")
print("• Network quality")
print("• Churn (Yes/No)")

# ============================================================================
# Step 3: Data Quality Checks
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: Data Quality Checks")
print("=" * 60)

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)
if missing_values.sum() == 0:
    print("[OK] No missing values")

duplicates = df.duplicated().sum()
print(f"\nDuplicates: {duplicates}")
if duplicates == 0:
    print("[OK] No duplicates")

# ============================================================================
# Step 4: Data Preparation
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: Data Preparation")
print("=" * 60)

df['signup_date'] = pd.to_datetime(df['signup_date'])
df['monthly_revenue'] = df['monthly_fee']
print("\n[OK] Converted signup_date to datetime")
print("[OK] Created monthly_revenue column")

# ============================================================================
# Step 5: Exploratory Data Analysis (EDA)
# ============================================================================
print("\n" + "=" * 60)
print("Step 5: Exploratory Data Analysis (EDA)")
print("=" * 60)

# Revenue by Subscription Plan
print("\n[PLOT] Revenue by Subscription Plan")
revenue_by_plan = df.groupby('subscription_plan')['monthly_revenue'].sum().sort_values(ascending=False)
print(revenue_by_plan)

plt.figure(figsize=(10, 6))
revenue_by_plan.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title("Monthly Revenue by Subscription Plan - TNM", fontsize=14, fontweight='bold')
plt.ylabel("Revenue (MWK)", fontsize=12)
plt.xlabel("Subscription Plan", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('tnm_revenue_by_plan.png', dpi=300, bbox_inches='tight')
print("[OK] Saved plot: tnm_revenue_by_plan.png")
plt.show()

print("\nInsight: Unlimited plans generate the highest revenue.")

# Data Usage vs Churn
print("\n[PLOT] Data Usage vs Churn")
plt.figure(figsize=(10, 6))
sns.boxplot(x='churned', y='data_usage_gb', data=df, palette=['#ff7f0e', '#2ca02c'])
plt.title("Data Usage vs Churn - TNM", fontsize=14, fontweight='bold')
plt.xlabel("Churned", fontsize=12)
plt.ylabel("Data Usage (GB)", fontsize=12)
plt.tight_layout()
plt.savefig('tnm_data_usage_vs_churn.png', dpi=300, bbox_inches='tight')
print("[OK] Saved plot: tnm_data_usage_vs_churn.png")
plt.show()

print("\nInsight: Customers with low data usage are more likely to churn.")

# Network Downtime by Churn
print("\n[PLOT] Network Downtime vs Churn")
plt.figure(figsize=(10, 6))
sns.barplot(x='churned', y='network_downtime_hours', data=df, palette=['#ff7f0e', '#2ca02c'])
plt.title("Network Downtime vs Churn - TNM", fontsize=14, fontweight='bold')
plt.xlabel("Churned", fontsize=12)
plt.ylabel("Network Downtime (Hours)", fontsize=12)
plt.tight_layout()
plt.savefig('tnm_downtime_vs_churn.png', dpi=300, bbox_inches='tight')
print("[OK] Saved plot: tnm_downtime_vs_churn.png")
plt.show()

print("\nInsight: Higher downtime = higher churn [WARNING]")

# ============================================================================
# Step 6: Key Business Questions Answered
# ============================================================================
print("\n" + "=" * 60)
print("Step 6: Key Business Questions Answered")
print("=" * 60)

# Which region churns the most?
print("\n[QUESTION] Which region churns the most?")
churn_by_region = df[df['churned'] == 'Yes']['region'].value_counts()
print(churn_by_region)

plt.figure(figsize=(10, 6))
churn_by_region.plot(kind='bar', color='#d62728')
plt.title("Churn by Region - TNM", fontsize=14, fontweight='bold')
plt.ylabel("Number of Churned Customers", fontsize=12)
plt.xlabel("Region", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('tnm_churn_by_region.png', dpi=300, bbox_inches='tight')
print("[OK] Saved plot: tnm_churn_by_region.png")
plt.show()

print("\n[FINDING] Southern region has higher churn.")

# Which plan is most stable?
print("\n[QUESTION] Which plan is most stable?")
plan_churn = df.groupby('subscription_plan')['churned'].value_counts(normalize=True)
print(plan_churn)

# Create a better visualization
churn_rates = df.groupby('subscription_plan')['churned'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print("\nChurn Rates by Plan (%):")
print(churn_rates)

plt.figure(figsize=(10, 6))
churn_rates.sort_values().plot(kind='barh', color='#2ca02c')
plt.title("Churn Rate by Subscription Plan - TNM", fontsize=14, fontweight='bold')
plt.xlabel("Churn Rate (%)", fontsize=12)
plt.ylabel("Subscription Plan", fontsize=12)
plt.tight_layout()
plt.savefig('tnm_churn_by_plan.png', dpi=300, bbox_inches='tight')
print("[OK] Saved plot: tnm_churn_by_plan.png")
plt.show()

print("\n[FINDING] Unlimited plan has the lowest churn rate.")

# ============================================================================
# Step 7: Business Insights (Very Important)
# ============================================================================
print("\n" + "=" * 60)
print("Step 7: Business Insights (Very Important)")
print("=" * 60)

print("\n[INSIGHTS] Key Insights:")
print("1. High network downtime leads to churn")
print("   - Average downtime for churned customers:", 
      f"{df[df['churned'] == 'Yes']['network_downtime_hours'].mean():.2f} hours")
print("   - Average downtime for active customers:", 
      f"{df[df['churned'] == 'No']['network_downtime_hours'].mean():.2f} hours")

print("\n2. Unlimited plans drive revenue and retention")
print("   - Revenue from Unlimited plans:", f"MWK {revenue_by_plan['Unlimited']:,.0f}")
print("   - Churn rate for Unlimited plans:", f"{churn_rates['Unlimited']:.2f}%")

print("\n3. Low-usage customers are churn-risk customers")
print("   - Average usage for churned customers:", 
      f"{df[df['churned'] == 'Yes']['data_usage_gb'].mean():.2f} GB")
print("   - Average usage for active customers:", 
      f"{df[df['churned'] == 'No']['data_usage_gb'].mean():.2f} GB")

print("\n4. Finding out which region needs network improvement")
print("   - Southern region has", churn_by_region.get('Southern', 0), "churned customers")
print("   - Average downtime in Southern region:", 
      f"{df[df['region'] == 'Southern']['network_downtime_hours'].mean():.2f} hours")

# ============================================================================
# Step 8: Recommendations (What TNM Wants to Hear)
# ============================================================================
print("\n" + "=" * 60)
print("Step 8: Recommendations (What TNM Wants to Hear)")
print("=" * 60)

print("\n[RECOMMENDATIONS] Strategic Recommendations:")
print("\n1. Improve network stability in Southern region")
print("   - Prioritize network infrastructure upgrades")
print("   - Reduce downtime to match other regions")

print("\n2. Target Basic & Standard plan users with upgrade offers")
print("   - Create compelling migration paths to Premium/Unlimited")
print("   - Offer promotional pricing for plan upgrades")

print("\n3. Monitor downtime KPIs closely")
print("   - Implement real-time monitoring dashboard")
print("   - Set alert thresholds for downtime incidents")
print("   - Establish rapid response protocols")

print("\n4. Introduce loyalty incentives for low-usage customers")
print("   - Reward customers with data bonuses")
print("   - Create engagement campaigns to increase usage")
print("   - Offer personalized plan recommendations")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)

