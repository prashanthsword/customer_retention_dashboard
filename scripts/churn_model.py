import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load the processed RFM data
df = pd.read_csv("data/processed_churn_data.csv")

# Define features and label
X = df[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']]
y = df['IsChurned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🎯 ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(model, "data/churn_model.pkl")
print("\n✅ Churn model saved to: data/churn_model.pkl")
