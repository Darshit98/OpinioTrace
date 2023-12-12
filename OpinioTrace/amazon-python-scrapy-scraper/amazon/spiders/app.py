import streamlit as st
import re
import subprocess
import boto3
import os
import pandas as pd
import time
import openai
from dotenv import load_dotenv
from transformers import pipeline
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import matplotlib.pyplot as plt

load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')


def upload_file_to_s3(file_name, s3_file_name):
    s3_client = boto3.client(
        service_name='s3',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
    )
    
    response = s3_client.upload_file(file_name, os.getenv('S3_BUCKET_NAME'), s3_file_name)
    print(f'File uploaded to S3: {response}')

def read_file_from_s3(s3_file_name):
    s3 = boto3.resource(
        service_name='s3',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
    )
    obj = s3.Bucket(S3_BUCKET_NAME).Object(s3_file_name).get()
    #st.write(obj)
    df = pd.read_csv(obj['Body'], index_col=0)
    st.write(df)
    return df

def sentiment_analysis(text):
    try:
        # Initialize the classifier
        classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        
        # Truncate the text to 512 tokens
        truncated_text = text[:512]

        # Perform sentiment analysis
        result = classifier(truncated_text)
        return result[0]['label']
    except Exception as e:
        # Handle any exceptions
        return f"Error during sentiment analysis: {e}"

def plot_sentiment_over_time(df):
    """
    Plots sentiment trend over time.
    """
    try:
        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Map sentiment to numerical values
        sentiment_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)

        # Group by date and calculate mean sentiment score
        sentiment_over_time = df.groupby(df['date'].dt.date)['sentiment_score'].mean()

        # Plotting
        plt.figure(figsize=(12, 6))
        sentiment_over_time.plot(kind='line', color='blue')
        plt.title('Sentiment Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in plotting sentiment trend: {e}")

def clean_data(df):
    # Drop unnecessary columns
    #df = df.drop(['asin', 'verified'], axis=1)

    # Extract "location" and "date" from "location_and_date"
    df['location'] = df['location_and_date'].str.extract(r'Reviewed in (?:the )?([\w\s]+) on')
    df['date'] = pd.to_datetime(df['location_and_date'].str.extract(r'on (.+)', expand=False), errors='coerce')

    # Format "date" as MM/DD/YYYY
    df['date'] = df['date'].dt.strftime('%m/%d/%Y')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop the original "location_and_date" column
    df.drop('location_and_date', axis=1, inplace=True)

    # Drop duplicates pairs for 'text' and 'title' columns
    df.drop_duplicates(subset=['text', 'title'], keep='first', inplace=True)

    # Replace missing values in 'text' and 'title'
    df['text'].fillna('Not Available', inplace=True)
    df['title'].fillna('No Title', inplace=True)

    # Drop rows where 'rating' is missing
    df.dropna(subset=['rating'], inplace=True)

    # Combine 'title' and 'text' into a new column 'combined_text'
    df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    df.drop(['title', 'text'], axis=1, inplace=True)

    # Text cleaning directly applied to the combined_text column
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    df['combined_text'] = df['combined_text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'@\w+|#', '', x))
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    st.write('Cleaned Data')

    # Applying sentiment analysis to each row
    df['sentiment'] = df['combined_text'].apply(sentiment_analysis)


    return df


def crawl_amazon_reviews(product_code):
  subprocess.run(["scrapy", "crawl", "amazon_reviews", "-a", f"asin={product_code}"])
  #uploading file to S3
  upload_file_to_s3('data/amazon_reviews.csv', 'scrapedreviews.csv')

  # Reading the file from S3
  df = read_file_from_s3('scrapedreviews.csv')

  # Check if the DataFrame is not empty
  if not df.empty:
      # Clean the data
      cleaned_df = clean_data(df)
      # Display or further process the cleaned DataFrame
      st.write(cleaned_df)
      #plotting trend
      plot_sentiment_over_time(cleaned_df)
  else:
      st.error('No data available to display after reading from S3.')

# Create a session state for login status
if 'logged_in' not in st.session_state:
  st.session_state.logged_in = False

# # Create a basic login system
# def login():
#   username = st.text_input("Username", value="", type="default")
#   password = st.text_input("Password", value="", type="password")

#   if st.button('Login'):
#     if username == 'admin' and password == 'sentiment':
#       st.session_state.logged_in = True
#     else:
#       st.error('Wrong credentials')

# def main():
#   if not st.session_state.logged_in:
#     login()
#   else:
#     # Input for URL
#     url_input = st.text_input('Enter Amazon product URL:')

#     # Show the search button only when there's some value in the url_input
#     if url_input:

#         if st.button('Search'):
#           # Regular expression to match the product code
#           st.write(f"URL Input:"+str(url_input))

#           pattern = r'dp/(.+)/ref|\/dp\/([^\/]+)'   
#           match = re.search(pattern, url_input)  
#           if match:    
#             st.write(match.group(1) or match.group(2))
#             product_code = match.group(1) or match.group(2) 
#             st.write(f"Extracted product code: {product_code}")
#             crawl_amazon_reviews(product_code)
#           # If a match is found, extract the product code
          
#           else:
#               st.error('Unable to extract product code from the provided URL.')


# if __name__ == '__main__':
#   main()

# Function to handle login
def handle_login():
    username = st.session_state.username
    password = st.session_state.password
    if username == 'admin' and password == 'sentiment':
        st.session_state.logged_in = True
    else:
        st.error('Wrong credentials')

# Login UI
def login():
    st.title("Login")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.text_input("Username", key="username", on_change=handle_login)
        st.text_input("Password", key="password", type="password", on_change=handle_login)
        st.button('Login', on_click=handle_login)

# Main application
def main():
    if not st.session_state.logged_in:
        login()
    else:
        st.title("Product Review Analysis")
        url_input = st.text_input('Enter Amazon product URL:')
        
        if url_input:
            if st.button('Search'):
                # Regular expression to match the product code
                pattern = r'dp/(.+)/ref|\/dp\/([^\/]+)'   
                match = re.search(pattern, url_input)  
                if match:    
                    product_code = match.group(1) or match.group(2) 
                    st.write(f"Extracted product code: {product_code}")
                    # Call your function to process the product code
                    crawl_amazon_reviews(product_code)
                else:
                    st.error('Unable to extract product code from the provided URL.')

if __name__ == '__main__':
    main()