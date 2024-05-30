#%%
import os
from dotenv import load_dotenv
import requests
import sqlite3
import pickle

#%%
# Define the setup function to load SVM model and TF-IDF vectorizer
def setup():
    # Load the SVM model
    with open("/work/MLops/svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("/work/MLops/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("/work/MLops/news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, cursor

# Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
svm_classifier, vectorizer, cursor = setup()

#%%
dotenv_path = '/work/MLops/.env'
load_dotenv(dotenv_path)

#%%
# Function to predict sentiment for a given title using the SVM model
def predict_sentiment_svm(title):
    # Vectorize the title
    title_vectorized = vectorizer.transform([title])

    # Make prediction using SVM classifier
    prediction = svm_classifier.predict(title_vectorized)

    # Return the predicted sentiment (capitalized)
    return prediction[0].capitalize()

#%%
# Function to predict sentiment, political leaning, and bias for a given title using the Together model
def predict_sentiment_together(title):
    # Define the prompt for the Together API
    prompt = f"""\
Label the news headline as either 'Positive', 'Negative', or 'Neutral', and indicate if the sentence is biased (minimal biased, moderate biased or not applicable) and what the political leaning is (Left-leaning, Right-leaning, Centrist, Not applicable):

Headline: {title}
Sentiment: 
Bias: 
Political Leaning: 
"""

    # Make request to Together API
    endpoint = 'https://api.together.xyz/inference'
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    res = requests.post(endpoint, json={
        "model": 'meta-llama/Llama-3-8b-chat-hf',
        "prompt": prompt,
        "top_p": 1,
        "top_k": 40,
        "temperature": 0.8,
        "max_tokens": 50,  # Increased to handle longer responses
        "repetition_penalty": 1,
    }, headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "User-Agent": "<YOUR_APP_NAME>"
    })

    # Extract sentiment, bias, and political leaning from response
    response_choices = res.json()['output']['choices']
    for choice in response_choices:
        text = choice['text'].strip()

        # Extract Sentiment
        if "Positive" in text:
            sentiment = "Positive"
        elif "Negative" in text:
            sentiment = "Negative"
        elif "Neutral" in text:
            sentiment = "Neutral"
        else:
            sentiment = "Unknown"

        # Extract Bias
        if "Minimal bias" in text:
            bias = "Minimal bias"
        elif "Moderate bias" in text:
            bias = "Moderate bias"
        elif "Not applicable" in text:
            bias = "Not applicable"
        else:
            bias = "Unknown"

        # Extract Political Leaning
        if "Left-leaning" in text:
            political_leaning = "Left-leaning"
        elif "Right-leaning" in text:
            political_leaning = "Right-leaning"
        elif "Centrist" in text:
            political_leaning = "Centrist"
        elif "Not applicable" in text:
            political_leaning = "Not applicable"
        else:
            political_leaning = "Unknown"

            if sentiment != "Unknown" and bias != "Unknown" and political_leaning != "Unknown":
                break  # Exit loop if all values are found

    return sentiment, bias, political_leaning


#%%
# Function to fetch the 10 newest articles and predict sentiment using both models
def predict_sentiment_for_newest_articles(max_articles=10):
    # List to store sentiment predictions
    all_articles_sentiments = []

    # Fetch the newest articles from the database
    cursor.execute("SELECT title FROM news_items ORDER BY published_at DESC LIMIT ?", (max_articles,))
    articles = cursor.fetchall()

    # Iterate over each article and predict sentiment using both models
    for article in articles:
        title = article[0]
        sentiment_svm = predict_sentiment_svm(title)
        sentiment_together, bias, political_leaning = predict_sentiment_together(title)
        all_articles_sentiments.append((title, sentiment_svm, sentiment_together, bias, political_leaning))

    return all_articles_sentiments

#%%
# Predict sentiment for the newest articles using both models
articles_sentiments = predict_sentiment_for_newest_articles()

#%%
# Print sentiment predictions
for title, sentiment_svm, sentiment_together, bias, political_leaning in articles_sentiments:
    print("Title:", title)
    print("Sentiment (SVM):", sentiment_svm)
    print("Sentiment (Together):", sentiment_together)
    print("Bias (Together):", bias)
    print("Political Leaning (Together):", political_leaning)
    print()

# %%
