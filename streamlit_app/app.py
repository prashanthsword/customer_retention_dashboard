import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Retention Dashboard", layout="wide")

st.title("📊 Customer Retention Dashboard")

# Load cleaned RFM data and model
df = pd.read_csv("data/processed_churn_data.csv")
model = joblib.load("data/churn_model.pkl")

# Sidebar filter
segment_filter = st.sidebar.selectbox("Select Segment", ["All"] + sorted(df["Segment"].unique().tolist()))
if segment_filter != "All":
    df = df[df["Segment"] == segment_filter]

st.subheader(f"Showing segment: **{segment_filter}** ({len(df)} customers)")

# Show metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average Recency (days)", f"{df['Recency'].mean():.0f}")
col2.metric("Avg Frequency", f"{df['Frequency'].mean():.1f}")
col3.metric("Churn Rate", f"{100 * df['IsChurned'].mean():.1f}%")

st.markdown("## 📈 RFM Feature Distributions")

plot_cols = st.columns(3)

with plot_cols[0]:
    st.image("data/recency_dist.png", caption="Recency", use_container_width=True)

with plot_cols[1]:
    st.image("data/frequency_dist.png", caption="Frequency", use_container_width=True)

with plot_cols[2]:
    st.image("data/monetary_dist.png", caption="Monetary", use_container_width=True)

st.markdown("## 🤖 Churn Probability per Customer")

# Predict churn probability using ML model
X = df[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']]
df['Churn_Prob'] = model.predict_proba(X)[:, 1]

# Highlight potential churners
def highlight_churn(val):
    return 'background-color: #ff4d4d; color: white' if val > 0.5 else ''

st.dataframe(df[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Churn_Prob']].style.applymap(highlight_churn, subset=['Churn_Prob']))
