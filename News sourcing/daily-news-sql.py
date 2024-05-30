#%%
import requests
import re
import pandas as pd
#import chromadb
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# %%
load_dotenv(dotenv_path= "/work/MLops/.env")
api_key = os.getenv("NEWS_API_KEY")

# %%
# Function to fetch data from API for the previous day
def fetch_data_from_api(api_key):
    # Calculate the date for the previous day
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    api_url = f"https://newsapi.org/v2/everything?q=*&from={previous_day}&to={previous_day}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        print("Failed to fetch data from API.")
        return None

# %%
# Define the class containing the article retrieval functions
class ArticleRetriever:
    def get_BuinessInsider(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')

            script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
            json_data = json.loads(script_tag.string)
            body = json_data['props']['pageProps']['articleShowData']['body']

            soup = BeautifulSoup(body, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_Forbes(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', class_='article-body fs-article fs-responsive-text current-article')

            child_tags = target_element.find_all(['h2', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_Android_Central(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', id='article-body')

            child_tags = target_element.find_all(['h2', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_Gizmodo_com(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', class_='sc-xs32fe-0 gKylik js_post-content')

            child_tags = target_element.find_all(['h3', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_bbc_news(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            soup.findAll("p", {"class": "ssrcss-1q0x1qg-Paragraph eq5iqo00"})
            soup.findAll("div", {"data-component": "text-block"})
            for para in soup.findAll("div", {"data-component": "text-block"}):
                news.append(para.find("p").getText())
            joinnews = " ".join(news)

            return joinnews
        except:
            return "None"

    def get_al_jazeera_english(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")

            paragraphs = soup.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_allafrica(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            target_element = soup.find('div', class_="story-body")
            paragraphs = target_element.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_abc_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'data-testid': 'prism-article-body'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_Globalsecurity_org(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'main'}).find('div', {'id': 'content'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_rt(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article").find('div', class_="article__text text")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_market_screener(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="txt-s4 article-text article-text--clear")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            meta_tag = soup.find('meta', {'http-equiv': 'refresh'})
            pattern = re.compile(r'content="0;url=\'(.*?)\'" http-equiv="refresh"')
            match = pattern.search(str(meta_tag)).group(1)
            web_url = "https://www.marketscreener.com{}".format(match)
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", web_url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="txt-s4 article-text article-text--clear ")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        finally:
            return "None"

    def get_phys_org(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="mt-4 article-main")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            splited = text_content.split("More information:")
            text_content = splited[0]

            return text_content

        except:
            return "None"

    def get_time_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'article-body-main'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_npr(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'storytext'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_boing_boing(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('section', class_="entry-content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_cna(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="text").find('div', class_="text-long")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_punch(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="post-content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_euronews(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            script_tag = soup.find('script', {'type': 'application/ld+json'})
            json_data = json.loads(script_tag.string)["@graph"][0]["articleBody"]
            return json_data

        except:
            return "None"

    def get_dedline_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div',
                                    class_="a-content pmc-u-line-height-copy pmc-u-font-family-georgia pmc-u-font-size-16 pmc-u-font-size-18@desktop")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_readwrite(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="entry-content col-md-10")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs[:-1]])
            return text_content

        except:
            return "None"

    def get_international_buiness_times(self, url):
        try:
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article-paywall-contents")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_cnn(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article__content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_The_Verge(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")

            target_element = soup.find('div',
                                       class_='duet--article--article-body-component-container clearfix sm:ml-auto md:ml-100 md:max-w-article-body lg:mx-100')
            text_content = target_element.get_text(separator=' ', strip=True)

            return text_content
        except:
            return "None"

    def get_indian_express(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="story_details")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_wired(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="body__inner-container")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_global_news_wire(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_etf_daily_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs[:-1]])

            return text_content

        except:
            return "None"

    def get_times_of_india(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="_s30J clearfix")

            all_text = content_div.text
            return all_text

        except:
            try:
                news_response = requests.get(url)
                soup = BeautifulSoup(news_response.content, features="html.parser")
                content_div = soup.select('article[class^="artData clr"]')
                all_text = '\n'.join([div.get_text(separator=' ') for div in content_div])
                return all_text
            except:
                return "None"

    def get_digital_content(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('article', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"
    def preprocess(self, source_name, url):
    
            if source_name == "Business Insider":
                return self.get_BuinessInsider(url)
    
            if source_name == "Forbes":
                return self.get_Forbes(url)
    
            if source_name == "Android Central":
                return self.get_Android_Central(url)
    
            if source_name == "Gizmodo.com":
                return self.get_Gizmodo_com(url)
    
            if source_name == "BBC News":
                return self.get_bbc_news(url)
    
            if source_name == "Al Jazeera English":
                return self.get_al_jazeera_english(url)
    
            if source_name == "AllAfrica - Top Africa News":
                return self.get_allafrica(url)
    
            if source_name == "ABC News":
                return self.get_abc_news(url)
    
            if source_name == "Globalsecurity.org":
                return self.get_Globalsecurity_org(url)
    
            if source_name == "RT":
                return self.get_rt(url)
    
            if source_name == "Marketscreener.com":
                return self.get_market_screener(url)
    
            if source_name == "Phys.Org":
                return self.get_phys_org(url)
    
            if source_name == "Time":
                return self.get_time_news(url)
    
            if source_name == "NPR":
                return self.get_npr(url)
    
            if source_name == "Boing Boing":
                return self.get_boing_boing(url)
    
            if source_name == "CNA":
                return self.get_cna(url)
    
            if source_name == "The Punch":
                return self.get_punch(url)
    
            if source_name == "Euronews":
                return self.get_euronews(url)
    
            if source_name == "Deadline":
                return self.get_dedline_news(url)
    
            if source_name == "ReadWrite":
                return self.get_readwrite(url)
    
            if source_name == "International Business Times":
                return self.get_international_buiness_times(url)
    
            if source_name == "CNN":
                return self.get_cnn(url)
    
            if source_name == "The Verge":
                return self.get_The_Verge(url)
    
            if source_name == "The Indian Express":
                return self.get_indian_express(url)
    
            if source_name == "Wired":
                return self.get_wired(url)
    
            if source_name == "GlobeNewswire":
                return self.get_global_news_wire(url)
    
            if source_name == "ETF Daily News":
                return self.get_etf_daily_news(url)
    
            if source_name == "The Times of India":
                return self.get_times_of_india(url)
    
            if source_name == "Digital Trends":
                return self.get_digital_content(url)
    
            else:
                return "None"


# %%
def preprocess_data(data, article_retriever):
    df = pd.DataFrame(data)
    df = df[['source', 'title', 'description', 'url', 'urlToImage', 'publishedAt', 'content']]
    
    # Filter out articles marked as '[Removed]'
    df = df[df['title'] != '[Removed]']
    
    # Rename columns
    df = df.rename(columns={'source': 'Name'})
    
    # Insert ID column
    df.insert(0, 'ID', range(1, 1 + len(df)))
    
    # Handle NULL values in the 'urlToImage' column by setting a default value
    df['urlToImage'] = df['urlToImage'].fillna('No Image url found')
    
    # Process 'Content' column, retrieving content if it's NULL
    df['Content'] = df.apply(lambda row: article_retriever.preprocess(row['Name']['name'], row['url']) if pd.isna(row['content']) else row['content'], axis=1)
    
    # Convert 'publishedAt' to datetime
    df['publishedAt'] = df['publishedAt'].apply(lambda x: datetime.fromisoformat(x.replace('Z', '+00:00')))
    
    return df

# %%
article_retriever = ArticleRetriever()
# Fetch data from API
api_data = fetch_data_from_api(api_key)
article_retriever = ArticleRetriever()
# Fetch data from API
api_data = fetch_data_from_api(api_key)
# Preprocess data
preprocessed_data = preprocess_data(api_data, article_retriever)

# %%
# Define the SQLAlchemy base
Base = sqlalchemy.orm.declarative_base()

# Define the NewsItem ORM model
class NewsItem(Base):
    __tablename__ = 'news_items'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    published_at = Column(DateTime)
    name = Column(String)
    content = Column(String)
    url = Column(String)
    url_to_image = Column(String)

# Connect to your SQL database 
engine = create_engine('sqlite:////work/MLops/news_database.db')
Base.metadata.create_all(engine)  # Create tables if they don't exist

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Iterate over the DataFrame and insert data into the database
for title, published_at, name, content, url, url_to_image, id in zip(preprocessed_data['title'], preprocessed_data['publishedAt'], preprocessed_data['Name'], preprocessed_data['Content'], preprocessed_data['url'], preprocessed_data['urlToImage'], preprocessed_data['ID']):
    news_item = NewsItem(title=title, published_at=published_at, name=name['name'], content=content, url=url, url_to_image=url_to_image)
    session.add(news_item)

# Commit the transaction
session.commit()
