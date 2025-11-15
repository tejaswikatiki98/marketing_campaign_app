# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression   # You can replace this with SVC, RandomForest, etc.
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"C:\Users\DELL\Documents\marketing_dashboard\marketing_campaign_realistic.csv")
# Separate features and target
X = df.drop('Response', axis=1)
y = df['Response']
# Define categorical and numeric columns
categorical = [
    "Gender",
    "Marital_Status",
    "Occupation",
    "Product_Category",
    "Preferred_Channel"
]
numeric = [col for col in X.columns if col not in categorical]
# Build full pipeline (preprocessing + model)
model = Pipeline([
    ('preprocess', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numeric)
    ], remainder='drop')),
    ('clf', RandomForestClassifier(n_estimators=20))
])
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit and predict
model.fit(X_train, y_train)
ypred = model.predict(X_test)
# Evaluate
print("Accuracy:", accuracy_score(y_test, ypred))
print(classification_report(y_test, ypred))


import joblib

# Save your trained model (with preprocessing pipeline)
joblib.dump(model, "marketing_response_model.pkl")

print("✅ Model saved successfully!")

