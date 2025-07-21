# Intelligent Document Classifier for SMEs
# Author: Jacques-Edgard Bossou
# Description: A simple machine learning script that classifies short text documents into business-relevant categories.

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import classification_report 

# Sample data - categories: Finance, Tech, Marketing
data = {
    'text': [
        "The invoice is due by end of month.",
        "Deploy the new update to the backend server.",
        "Our next campaign should focus on social media engagement.",
        "Quarterly earnings exceeded expectations.",
        "Fix the bug in the login function.",
        "Let's design banners for the new product launch.",
        "The budget needs to be revised before submission.",
        "Set up the staging environment for testing.",
        "We should increase SEO spend next quarter."
    ],
    'label': [
        "Finance",
        "Tech",
        "Marketing",
        "Finance",
        "Tech",
        "Marketing",
        "Finance",
        "Tech",
        "Marketing"
    ]
}

# Load data into DataFrame
df = pd.DataFrame(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Create a pipeline: TF-IDF Vectorizer + Naive Bayes Classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --- Simple CLI interface ---
print("\nTry the classifier (type 'exit' to quit):")
while True:
    user_input = input("\nEnter a short business text: ")
    if user_input.lower() == 'exit':
        break
    prediction = model.predict([user_input])[0]
    print(f"Predicted Category: {prediction}")
