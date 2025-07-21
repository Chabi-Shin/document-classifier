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
        "Process payroll before the weekend.",
        "The bank reconciliation is incomplete.",
        "Audit the expense reports for discrepancies.",
        "Forecast next year's revenue growth.",
        "Approval needed for capital expenditure.",
        "The CFO wants a cost-benefit analysis.",
        "Update the financial projections.",
        "Track departmental spending this quarter.",
        "File the quarterly tax returns.",
        "Review the investment portfolio performance.",
        "The credit line needs to be extended.",
        "Calculate ROI for the latest campaign.",
        "The accounting software needs an upgrade.",
        "Reconcile the petty cash transactions.",
        "The shareholders meeting is next week.",
        "Prepare the annual financial statements.",
        "The invoice payment is overdue.",
        "Analyze cash flow trends for Q4.",
        "The budget proposal was rejected.",
        "The audit committee raised concerns.",
        "The VAT filing deadline is approaching.",
        "The treasurer needs expense reports.",
        "The financial model needs adjustments.",
        "The loan application was approved.",
        "The cost-cutting measures are working.",
        "The profit margin has improved.",
        "The financial dashboard needs updates.",
        "The accounts payable are delayed.",
        "The revenue recognition policy changed.",
        "The tax consultant provided new advice.",
        "The depreciation schedule is outdated.",
        "The financial risk assessment is due.",
        "The merger requires due diligence.",
        "The investor relations team needs data.",
        "The dividend payout is scheduled.",
        "The financial compliance audit passed.",
        "The credit card statements need review.",
        "The financial analyst requested more data.",
        "The hedge fund performance declined.",
        "The stock options vesting is pending.",

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
        "The CI/CD pipeline needs optimization.",
        "The Kubernetes cluster is over-provisioned.",
        "The machine learning model needs retraining.",
        "The frontend UI is not responsive.",
        "The cybersecurity audit revealed vulnerabilities.",
        "The data warehouse needs restructuring.",
        "The dev team is switching to React.",
        "The API rate limits are too restrictive.",
        "The blockchain integration is in progress.",
        "The AI chatbot needs more training data.",
        "The DNS configuration is incorrect.",
        "The load balancer is under heavy traffic.",
        "The microservices architecture is scaling well.",
        "The SQL query performance is slow.",
        "The IoT devices need firmware updates.",
        "The VPN connection keeps dropping.",
        "The test coverage is below 80%.",
        "The container registry is full.",
        "The encryption keys need rotation.",
        "The big data pipeline is failing.",
        "The webhooks are not triggering correctly.",
        "The GraphQL schema needs updates.",
        "The NoSQL database is running out of space.",
        "The edge computing setup is complete.",
        "The AR/VR project is behind schedule.",
        "The quantum computing research is ongoing.",
        "The robotics team needs more sensors.",
        "The 5G network rollout is delayed.",
        "The autonomous vehicle algorithm needs tuning.",
        "The voice recognition accuracy improved.",
        "The SaaS platform had a major outage.",
        "The dark mode feature is now live.",
        "The password reset flow is broken.",
        "The data privacy compliance is pending.",
        "The open-source contribution was accepted.",
        "The neural network training is complete.",
        "The Wi-Fi signal strength is weak.",
        "The backup restoration failed.",
        "The IDE plugin needs an update.",
        "The GitHub repository was migrated.",

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
        "Improve the landing page conversion rate.",
        "The customer engagement metrics are rising.",
        "The A/B test results are inconclusive.",
        "The PR team drafted a press release.",
        "The competitor analysis is complete.",
        "The brand awareness survey is live.",
        "The Google Ads budget needs adjustment.",
        "The TikTok campaign went viral.",
        "The webinar registration is low.",
        "The affiliate marketing program is expanding.",
        "The customer persona needs refinement.",
        "The lead generation form isn't converting.",
        "The retargeting ads are performing well.",
        "The content calendar is outdated.",
        "The video marketing strategy is working.",
        "The newsletter open rates dropped.",
        "The market research data is ready.",
        "The trade show booth design is approved.",
        "The referral program needs promotion.",
        "The customer testimonials are compelling.",
        "The LinkedIn engagement is increasing.",
        "The YouTube channel needs more content.",
        "The hashtag strategy isn't effective.",
        "The brand guidelines need updating.",
        "The sales funnel has a leakage issue.",
        "The CRM data needs segmentation.",
        "The promotional video is in editing.",
        "The UX copy needs optimization.",
        "The podcast sponsorship is confirmed.",
        "The loyalty program is launching soon.",
        "The PPC campaign ROI is positive.",
        "The offline marketing efforts are lagging.",
        "The meme marketing experiment succeeded.",
        "The customer journey map is incomplete.",
        "The geotargeting settings are incorrect.",
        "The omnichannel strategy is paying off.",
        "The user-generated content is valuable.",
        "The brand sentiment analysis is positive.",
        "The Facebook group engagement is high.",
        "The seasonal campaign planning starts now.",
        "The marketing automation is saving time."
    ],
    'label': [
        # FINANCE (50)
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",
        "Finance", "Finance", "Finance", "Finance", "Finance",

        # TECH (50)
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",
        "Tech", "Tech", "Tech", "Tech", "Tech",

        # MARKETING (50)
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
        "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
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
