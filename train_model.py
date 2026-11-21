
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

data = data.dropna()
data["label"] = data["label"].str.lower()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Create Logistic Regression model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model: Logistic Regression")
print("Accuracy:", accuracy * 100, "%")

# Save model
joblib.dump(model, "spam_model.pkl")

print("Model trained and saved successfully!")
