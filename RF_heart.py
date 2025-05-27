import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load and clean the data
df = pd.read_csv("/Users/anamikaroy/Desktop/practice/ML/framingham_heart_disease.csv")
df = df.dropna()

# Define input and output
X = df.iloc[:, :15]
Y = df["TenYearCHD"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=27)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=27)
rf_model.fit(X_train_scaled, Y_train)

# Predictions
Y_pred = rf_model.predict(X_test_scaled)
Y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Confusion Matrix:")
cm = metrics.confusion_matrix(Y_test, Y_pred)
print(cm)

print("\nClassification Report:")
print(metrics.classification_report(Y_test, Y_pred))

print("ROC AUC Score:", metrics.roc_auc_score(Y_test, Y_prob))

# Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# Feature importance plot
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
