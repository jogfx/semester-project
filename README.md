# Semester project - 2nd semester Business Data Science

[Link to Streamlit app](https://app-project-mlops.cloud.sdu.dk/)

# News Media Analysis Project

## Introduction
In today's world, news outlets hold significant power to influence society, impacting individuals, organizations, and businesses. They shape public opinion and control narratives through selective coverage. This project aims to create a tool leveraging Natural Language Processing (NLP) through traditional machine learning models and Large Language Models (LLMs) to analyze news media for sentiment, bias, and political leaning. The goal is to provide actionable insights for businesses to make informed decisions.

## Project Overview
This project combines traditional machine learning techniques with advanced LLMs to predict news sentiment, bias, and political leaning. By utilizing both approaches, we aim to enhance the reliability and accuracy of the predictions.

### Research Question
How can natural language processing through traditional machine learning and large language models be leveraged by businesses to analyze news media for sentiment, bias, and political leaning in order to gain actionable insights and make informed decisions?

## Literature Review
The literature review focuses on:
- The role of news media in shaping societies.
- The impact of digital transformation and the challenges posed by misinformation and bias.
- The potential of LLMs like ChatGPT and traditional models like SVMs for sentiment analysis and combating misinformation.

## Methodology

### Philosophical Framework
**Critical Realism** is the chosen philosophical foundation, emphasizing underlying mechanisms and deeper structures in news content. This approach ensures models capture cultural and temporal contexts within data.

### Scope of Analysis
- **Sentiment Analysis:** Classifying news articles as "Positive," "Negative," or "Neutral."
- **Political Orientation:** Labeling as "Liberal," "Conservative," or "Neutral."
- **Bias Detection:** Identifying unfair or unbalanced presentations in news headlines.

## Project Architecture

### Components
1. **Business Aspect:** Defines the problem and goal of using data science techniques to address the impact of misinformation and negative sentiment on businesses.
2. **Data Aspect:** Focuses on data acquisition, cleaning, preprocessing, and feature engineering.
3. **MLOps Aspect:** Involves model development, evaluation, deployment, and monitoring.

### Data Pipeline
- **Data Collection:** Using NewsAPI.org and Kaggle datasets.
- **Preprocessing:** Cleaning text, tokenization, lemmatization, removing stop words, and TF-IDF processing.
- **Model Training:** Training SVM models and utilizing Llama 3 for sentiment analysis through instruction-based prompting.

## Evaluation Metrics
- **Accuracy:** Proportion of correctly classified instances.
- **Precision:** Ratio of true positive predictions to the sum of true and false positives.
- **Recall:** Ratio of true positive predictions to the sum of true positives and false negatives.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Describes the performance of the classification model.

## Model Performance

### SVM Classifier
- **Accuracy:** 84.5%
- **Challenges:** Struggles with class imbalance and identifying positive sentiments.

### Llama 3 8B Model
- **Accuracy:** 93% for sentiment analysis, 92% for bias analysis, and 94% for political leaning classification.
- **Performance:** Superior to SVM in understanding and predicting nuanced language aspects.

## Frontend Application
The application, built using Streamlit, allows users to:
- Select news sources and view the latest articles with sentiment analysis.
- Input custom headlines for analysis.
- View model performance metrics and sentiment distribution among articles.

## Future Implementations
- Utilize online SQL services like AWS or Google Cloud for reliability.
- Automate the data gathering process through an API wrapper.
- Break down the pipeline into microservices for scalability.
- Enhance the dataset with more diverse and manually labeled data for better model training and evaluation.

## Conclusion
This project demonstrates the potential of combining traditional machine learning models with LLMs for analyzing news media. The hybrid approach improves the reliability of sentiment, bias, and political leaning predictions, providing valuable insights for businesses. Further developments could focus on improving data quality, model performance, and scalability.
