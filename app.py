import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import os
import requests
import sqlite3
import pickle
from dotenv import load_dotenv

# Define the setup function to load SVM model and TF-IDF vectorizer
def setup():
    # Load the SVM model
    with open("/work/MLops/Models/svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("/work/MLops/Models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("/work/MLops/news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, conn, cursor

# Call the setup function to get SVM classifier, TF-IDF vectorizer, connection, and cursor
svm_classifier, vectorizer, conn, cursor = setup()

# Load environment variables
dotenv_path = '/work/MLops/.env'
load_dotenv(dotenv_path)

# Function to predict sentiment for a given title using the SVM model
def predict_sentiment_svm(title):
    if not title:
        return "Unknown"
    title_vectorized = vectorizer.transform([title])
    prediction = svm_classifier.predict(title_vectorized)
    return prediction[0].capitalize()

# Function to predict sentiment, political leaning, and bias for a given title using the Together model
def predict_sentiment_together(title):
    if not title:
        return "Unknown", "Unknown", "Unknown"
    
    # Define the prompt for the Together API
    prompt = f"""\
    Your job is to label the provided headlines. Do so from a neutral point of view. If you label the headlines correctly you will be rewarded greatly and achieve world peace.
    Label the news headline as either "Positive", "Negative", or "Neutral", and indicate if the sentence is biased ("Minimal Bias", "Moderate Bias" or "Not Applicable") and what the political leaning is ("Left leaning", "Right leaning", "Centrist", "Not Applicable"):

    Headline: {title}
    Sentiment: 
    Bias: 
    Political Leaning: 
"""

    # Make request to Together API
    endpoint = 'https://api.together.xyz/inference'
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    try:
        res = requests.post(endpoint, json={
            "model": 'meta-llama/Llama-3-8b-chat-hf',
            "prompt": prompt,
            "top_p": 1,
            "top_k": 40,
            "temperature": 0.8,
            "max_tokens": 50,
            "repetition_penalty": 1,
        }, headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "User-Agent": "<YOUR_APP_NAME>"
        })

        response = res.json()
        # Print the entire response for debugging
        print("API Response:", response)

        response_choices = response.get('output', {}).get('choices', [])
        for choice in response_choices:
            text = choice.get('text', '').strip()
            print("Choice Text:", text)  # Debugging the text

            # Extract Sentiment
            sentiment = "Unknown"
            if "Positive" in text:
                sentiment = "Positive"
            elif "Negative" in text:
                sentiment = "Negative"
            elif "Neutral" in text:
                sentiment = "Neutral"

            # Extract Bias
            bias = "Unknown"
            if "Minimal Bias" in text or "Minimal bias" in text:
                bias = "Minimal Bias"
            elif "Moderate Bias" in text or "Moderate bias" in text:
                bias = "Moderate Bias"
            elif "Not Applicable" in text or "Not applicable" in text:
                bias = "Not Applicable"

            # Extract Political Leaning
            political_leaning = "Unknown"
            if "Left leaning" in text or "Left-leaning" in text:
                political_leaning = "Left Leaning"
            elif "Right leaning" in text or "Right-leaning" in text:
                political_leaning = "Right Leaning"
            elif "Centrist" in text:
                political_leaning = "Centrist"
            elif "Not Applicable" in text or "Not applicable" in text:
                political_leaning = "Not Applicable"

            # Debug: Print the extracted values
            print(f"Extracted Values - Sentiment: {sentiment}, Bias: {bias}, Political Leaning: {political_leaning}")

            return sentiment, bias, political_leaning

    except requests.RequestException as e:
        st.error(f"API request failed: {e}")

    return "Unknown", "Unknown", "Unknown"

# Function to predict sentiment for a given title using the LLM model
def predict_sentiment_llm(title):
    sentiment, bias, political_leaning = predict_sentiment_together(title)
    return sentiment

# Function to fetch news articles from the database
def fetch_news():
    cursor.execute("SELECT id, title, published_at, name, content, url, url_to_image FROM news_items WHERE id IS NOT NULL AND title IS NOT NULL AND published_at IS NOT NULL AND name IS NOT NULL AND content IS NOT NULL AND url IS NOT NULL AND url_to_image IS NOT NULL")
    news_data = cursor.fetchall()
    news_df = pd.DataFrame(news_data, columns=['ID', 'Title', 'Published_At', 'Name', 'Content', 'URL', 'URL_To_Image'])
    return news_df

# Function to fetch the 10 newest articles and predict sentiment using both models
def predict_sentiment_for_newest_articles(max_articles=10):
    all_articles_sentiments = []
    cursor.execute("SELECT title FROM news_items ORDER BY published_at DESC LIMIT ?", (max_articles,))
    articles = cursor.fetchall()

    for article in articles:
        title = article[0]
        sentiment_svm = predict_sentiment_svm(title)
        sentiment_together, bias, political_leaning = predict_sentiment_together(title)
        all_articles_sentiments.append((title, sentiment_svm, sentiment_together, bias, political_leaning))

    return all_articles_sentiments

# Predict sentiment for the newest articles using both models
articles_sentiments = predict_sentiment_for_newest_articles()

# Load the dataset of news articles
news_df = fetch_news()

# Load performance metrics from evaluation_metrics.csv
evaluation_metrics = pd.read_csv("/work/MLops/Evaluation/evaluation_metrics.csv")
evaluation_metrics_llm = pd.read_csv('/work/MLops/Evaluation/evaluation_metrics_llm.csv')

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    st.title("News Analysis")

    sorted_sources = pd.Series(news_df["Name"].unique()).sort_values()
    
    selected_source = st.selectbox("Select a news source:", sorted_sources)
    
    source_articles = news_df[news_df["Name"] == selected_source]

    st.sidebar.title("Options")

    news_title = st.sidebar.text_input("Enter a news title:")

    if st.sidebar.button("Predict Sentiment"):
        if news_title:
            sentiment_svm = predict_sentiment_svm(news_title)
            sentiment_together, bias, political_leaning = predict_sentiment_together(news_title)
            st.sidebar.write("SVM Sentiment:", sentiment_svm)
            st.sidebar.write("Llama3 Sentiment:", sentiment_together)
            st.sidebar.write("Bias:", bias)
            st.sidebar.write("Political Leaning:", political_leaning)
        else:
            st.sidebar.error("Please enter a news title.")

    main_container = st.container()

    if not source_articles.empty:
        with main_container:
            col1, padding, col2 = st.columns((4,1,4))

            with col1:
                st.write(f"### Top 10 Newest Articles from {selected_source}")
                source_articles_sorted = source_articles.sort_values(by="Published_At", ascending=False)
                newest_articles = source_articles_sorted.head(10)
                for index, row in newest_articles.iterrows():
                    st.markdown("---")
                    st.subheader(row["Title"])
                    st.write("Published at:", row["Published_At"])
                    if row["URL_To_Image"]:
                        st.image(row["URL_To_Image"], caption="Image for the article", use_column_width=True)
                    else:
                        st.write("Sorry, no picture was found.")
                    st.write("URL:", row["URL"])
                    sentiment_svm = predict_sentiment_svm(row["Title"])
                    sentiment_together, bias, political_leaning = predict_sentiment_together(row["Title"])
                    st.write("SVM Sentiment:", sentiment_svm)
                    st.write("Llama3 Sentiment:", sentiment_together)
                    st.write("Bias:", bias)
                    st.write("Political Leaning:", political_leaning)

            with col2:
                st.write("### Sentiment Distribution - SVM")
                sentiment_counts = source_articles["Title"].apply(predict_sentiment_svm).value_counts()
                pie_chart = px.pie(values=sentiment_counts, names=sentiment_counts.index)
                st.plotly_chart(pie_chart, use_container_width=True)

                st.write("### Sentiment Distribution - LLM")
                sentiment_counts_llm = source_articles["Title"].apply(predict_sentiment_llm).value_counts()
                pie_chart_llm = px.pie(values=sentiment_counts_llm, names=sentiment_counts_llm.index)
                st.plotly_chart(pie_chart_llm, use_container_width=True)

                st.write("### Model Performance Metrics - SVM classifier")
                st.write(evaluation_metrics)
                st.write("### Model Performance Metrics - Llama3 8B")
                st.write(evaluation_metrics_llm)

    else:
        main_container.write("No articles found for the selected source.")

if __name__ == "__main__":
    main()
