import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nitter import Nitter

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

important_words = {
    'no', 'not', 'nor', "don't", "doesn't", "didn't", "hadn't",
    "hasn't", "haven't", "isn't", "mightn't", "mustn't",
    "needn't", "shan't", "shouldn't", "wasn't", "weren't", 
    "won't", "wouldn't", "can't", "couldn't", "ain", "aren", "doesn",
    "didn", "hadn", "hasn", "haven", "isn", "mightn", "mustn", 
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "cannot", "never"
}

def predicate_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [
        word for word in text if word.lower() not in stopwords.words('english') and word.lower() in important_words
    ]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    #predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment==0 else "Positive"

@st.cache_resource
def intialize_scraper():
    return Nitter(log_level=1)

def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

def main():
    st.title("Twitter Sentiment Analysis")
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = intialize_scraper()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option=="Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predicate_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

        elif option == "Get tweets from user":
            username = st.text_input("Enter Twitter username")
            if st.button("Fetch Tweets"):
                tweets_data = scraper.get_tweets(username, mode='user', number=5)
                if 'tweets' in tweets_data:
                    for tweet in tweets_data['tweets']:
                        tweet_text = tweet['text']
                        sentiment = predicate_sentiment(tweet_text, model, vectorizer, stop_words)
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.write("No tweets found or an error occurred.")

    if __name__ == "__main__":
        main()

