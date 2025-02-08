import pandas as pd
import numpy as np
import whois
import requests
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("malicious_phish.csv")

# Map categorical labels to numerical values
label_mapping = {"benign": 0, "phishing": 1, "defacement": 2}
df["label"] = df["type"].map(label_mapping)

def get_domain_age(domain):
    try:
        domain_whois = whois.whois(domain)
        creation_date = domain_whois.creation_date
        if isinstance(creation_date, list):  # Handle multiple dates case
            creation_date = creation_date[0]
        return (pd.Timestamp.now() - pd.to_datetime(creation_date)).days if creation_date else 0
    except whois.parser.PywhoisError:
        return 0  # Return 0 if WHOIS data is unavailable
    except Exception as e:
        print(f"WHOIS error for {domain}: {e}")
        return 0

def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.split(':')[0]  # Extract domain
    
    features = {
        'url_length': len(url),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_dots': url.count('.'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'https': 1 if parsed_url.scheme == 'https' else 0,
        'domain_age': get_domain_age(domain)
    }
    return features

# Apply Feature Extraction
df_features = df['url'].apply(lambda x: extract_features(x)).apply(pd.Series)

dataset = pd.concat([df_features, df['label']], axis=1)

# Split Data into Features (X) and Labels (y)
X = dataset.drop(columns=['label'])
y = dataset['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)

# Model Performance Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Function to Predict a Single URL
def predict_url(url):
    url_features = extract_features(url)
    features_df = pd.DataFrame([url_features])
    prediction = model.predict(features_df)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Test Example
test_url = "http://paypal-secure-login.com"
print(f"URL: {test_url} - Prediction: {predict_url(test_url)}")
