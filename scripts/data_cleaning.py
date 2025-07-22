import pandas as pd
from datetime import datetime

# Load the dataset
df = pd.read_csv("data/raw/synthetic_customer_data.csv")

# Preview
print("🧾 Raw data shape:", df.shape)
print(df.head())

# Convert date columns to datetime
df['JoinDate'] = pd.to_datetime(df['JoinDate'])
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])

# Assume today is our "analysis date"
today = pd.to_datetime("2025-07-22")

# Calculate Recency in days
df['Recency'] = (today - df['LastPurchaseDate']).dt.days

# Frequency is already 'OrdersCount'
# Monetary is already 'LifetimeValue'

# Build RFM DataFrame
rfm_df = df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'IsChurned']].copy()
rfm_df.rename(columns={
    'OrdersCount': 'Frequency',
    'LifetimeValue': 'Monetary'
}, inplace=True)

print("\n📊 RFM Table Preview:")
print(rfm_df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style="whitegrid")

# Distribution of Recency
plt.figure(figsize=(6, 4))
sns.histplot(rfm_df['Recency'], bins=10, kde=True)
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig("data/recency_dist.png")
plt.close()

# Frequency distribution
plt.figure(figsize=(6, 4))
sns.histplot(rfm_df['Frequency'], bins=10, kde=True, color='orange')
plt.title("Frequency Distribution")
plt.xlabel("Number of Orders")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig("data/frequency_dist.png")
plt.close()

# Monetary distribution
plt.figure(figsize=(6, 4))
sns.histplot(rfm_df['Monetary'], bins=10, kde=True, color='green')
plt.title("Monetary Distribution")
plt.xlabel("Total Spend")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig("data/monetary_dist.png")
plt.close()

print("✅ RFM distribution plots saved in /data folder.")

# Churn vs RFM stats
print("\n📉 Average RFM values for Churned vs Active customers:")
print(rfm_df.groupby("IsChurned")[['Recency', 'Frequency', 'Monetary']].mean())

# RFM SCORING (1–5 scale: 5 = best)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Combine RFM score
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

# Segment based on RFM_Score
def segment(row):
    if row['R_Score'] == 5 and row['F_Score'] >= 4:
        return 'VIP'
    elif row['R_Score'] >= 4:
        return 'Loyal'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
        return 'Lost'
    elif row['R_Score'] <= 3 and row['F_Score'] >= 3:
        return 'At Risk'
    else:
        return 'Others'

rfm_df['Segment'] = rfm_df.apply(segment, axis=1)

print("\n🏷️ Customer Segments Preview:")
print(rfm_df[['CustomerID', 'RFM_Score', 'Segment']])

rfm_df.to_csv("data/processed_churn_data.csv", index=False)
print("\n✅ Final cleaned dataset saved to: data/processed_churn_data.csv")
