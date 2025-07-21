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
        # FINANCE
        "Submit the invoice by the end of this week.",
        "The financial report is due tomorrow.",
        "Please review the Q3 earnings.",
        "We need to reduce the budget for next month.",
        "Prepare the tax documents for auditing.",
        "The balance sheet needs updating.",
        "Request a wire transfer for the vendor.",
        "The client is asking about payment terms.",
        "We exceeded last quarter's revenue target.",
        "The new pricing model needs approval.",

        # TECH
        "Deploy the backend code to production.",
        "Fix the bug in the authentication module.",
        "Update the system to the latest firmware.",
        "Schedule server maintenance this weekend.",
        "Implement the new API for user data.",
        "The database migration was successful.",
        "Set up the cloud infrastructure on AWS.",
        "Debug the mobile app crash issue.",
        "Integrate the payment gateway this sprint.",
        "Refactor the code for better performance.",

        # MARKETING
        "Launch the Instagram ad campaign tomorrow.",
        "Design banners for the upcoming product launch.",
        "Update the website SEO with fresh keywords.",
        "Our branding needs to target Gen Z more.",
        "Schedule email campaigns for next week.",
        "The social media reach has grown significantly.",
        "Prepare the slides for the marketing pitch.",
        "Let's work with influencers this quarter.",
        "Analyze the click-through rate from the ads.",
        "Improve the landing page conversion rate."
    ],
    'label': [
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing"
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
