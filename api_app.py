import streamlit as st
import pickle
import string
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.pexels.com/photos/1297790/pexels-photo-1297790.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Loading vectorizer and pretrained ML model 
cv = pickle.load(open('outputs/vectorizer.sav', 'rb'))
model = pickle.load(open('outputs/final_model.sav', 'rb'))

###################################################################
# Preprocessing tweets to remove any special characters or numbers
# for input to the model

lm = WordNetLemmatizer()

def tweets_cleaner(tweet):
    """Function to preprocess the tweets for the models

    Returns:
        tweet without stop words, special characters, RT, numbers,
        with stemming applied
    """
    # removing the urls from the text
    tweet = re.sub(r'https?://\S+', r'', tweet)
    tweet = re.sub(r'www.\S+', r'', tweet)
    # removing the numbers from the text
    tweet = re.sub(r'[0-9]\S+', r'', tweet)
    # removing the tags from the text
    tweet = re.sub(r'(@\S+) | (#\S+)', r'', tweet)
    # removing the RT from the text
    tweet = re.sub(r'\bRT\b', r'', tweet)
    # removing repeated characters
    tweet = re.sub(r'(.)1+', r'1', tweet)
    # applying tokenization
    tweet = re.split('\W+', tweet)
    # applying lemmatization
    tweet = [lm.lemmatize(word) for word in tweet]
    # removing the punctuation from the text
    tweet_without_punctuation = [char for char in tweet if char not   
                                in string.punctuation]
    # converting the list to string 
    tweet_without_punctuation = " ".join(tweet_without_punctuation) 
    # set of stop words 
    stop_words = set(stopwords.words("english"))
    # removing the stop words 
    tweet_without_stopwords = [word for word in  
                              tweet_without_punctuation.split()
                              if word.lower() not in stop_words]
    return " ".join(tweet_without_stopwords)
###################################################################


# Streamlit app
def main():
    st.title('Tweet Sentiment Analysis')

    # Get user input
    user_input = st.text_area("Enter your tweet:")

    if st.button("Predict"):
        # Preprocess the tweet
        processed_tweet = tweets_cleaner(user_input)

        # Vectorize and predict
        vectorized_tweet = cv.transform([processed_tweet])
        prediction = model.predict_proba(vectorized_tweet)
        prediction_label = 'Positive Sentiment' if prediction[0, 1] >= 0.5 else 'Negative Sentiment'

        # Display result
        st.write(f'Your tweet has a {prediction_label}.')

if __name__ == '__main__':
    main()
 
