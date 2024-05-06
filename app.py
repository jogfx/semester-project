import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import plotly.express as px
import pickle
import sqlite3

# Load your trained SVM model
with open("svm_classifier.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the dataset of news articles
#news_df = pd.read_csv("rating.csv")

# Connect to SQLite database
conn = sqlite3.connect("news_database.db")
cursor = conn.cursor()

# Function to preprocess and vectorize the input title
def preprocess_and_vectorize(title):
    preprocessed_title = vectorizer.transform([title])
    return preprocessed_title

# Function to predict sentiment
def predict_sentiment(title):
    preprocessed_title = preprocess_and_vectorize(title)
    sentiment_prediction = svm_classifier.predict(preprocessed_title)
    return sentiment_prediction

# Function to fetch news articles from the database
def fetch_news():
    cursor.execute("SELECT id, title, published_at, name, content, url, url_to_image FROM news_items")
    news_data = cursor.fetchall()
    news_df = pd.DataFrame(news_data, columns=['ID', 'Title', 'Published_At', 'Name', 'Content', 'URL', 'URL_To_Image'])
    return news_df

# Load the dataset of news articles
news_df = fetch_news()

# Load performance metrics from evaluation_metrics.csv
evaluation_metrics = pd.read_csv("evaluation_metrics.csv")

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    st.title("News Sentiment Analysis")

    # Sort news sources alphabetically
    sorted_sources = pd.Series(news_df["Name"].unique()).sort_values()
    
    # Dropdown to select sorted news source
    selected_source = st.selectbox("Select a news source:", sorted_sources)
    
    # Filter articles based on selected source
    source_articles = news_df[news_df["Name"] == selected_source]

    st.sidebar.title("Options")

    # Sidebar option to enter a news title
    news_title = st.sidebar.text_input("Enter a news title:")

    if st.sidebar.button("Predict Sentiment"):
        if news_title:
            prediction = predict_sentiment(news_title)
            if prediction == 1:
                st.sidebar.write("Sentiment: Positive")
            elif prediction == -1:
                st.sidebar.write("Sentiment: Negative")
            else:
                st.sidebar.write("Sentiment: Neutral")

    # Create a container for the main content
    main_container = st.container()

    if not source_articles.empty:
        with main_container:
            # Create two columns layout
            col1, padding, col2 = st.columns((4,1,4))

            # Display top 10 newest articles from selected source on the left column
            with col1:
                st.write("### Top 10 Newest Articles from", selected_source)
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
                    sentiment = predict_sentiment(row["Title"])
                    st.write("Sentiment:", sentiment[0])

            # Create a pie chart showing sentiment distribution on the right column
            with col2:
                st.write("### Sentiment Distribution")
                sentiment_counts = source_articles["Title"].apply(predict_sentiment).value_counts()
                pie_chart = px.pie(values=sentiment_counts, names=sentiment_counts.index)
                st.plotly_chart(pie_chart, use_container_width=True)

                # Display model performance metrics with two decimal places
                st.write("### Model Performance Metrics - SVM classifier")
                st.write(evaluation_metrics)

    else:
        main_container.write("No articles found for the selected source.")

if __name__ == "__main__":
    main()