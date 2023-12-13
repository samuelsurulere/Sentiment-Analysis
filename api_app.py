# importing necessary libraries
from flask import Flask, request, render_template
import pickle
import string
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
# from werkzeug.utils import secure_filename, redirect

# Create a Flask application
app = Flask(__name__, template_folder='templates')
 
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


# Create a home page
@app.route('/')
def home():
    """
    For displaying frontend HTML interface
    """
    return render_template('tweet_sentiment.html')

# Create a POST method
@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    # Extract the value of the 'sentence' form field
    # to_predict_list = request.form.to_dict()
    # review_text = pre_processing(to_predict_list['review_text'])
    
    # pred = clf.predict(count_vect.transform([review_text]))
    
    tweet = tweets_cleaner(request.form['sentence'])
    
    vectorized_tweet = cv.transform([tweet])
    prediction = model.predict_proba(vectorized_tweet)
    prediction = ['Positive Sentiment' if proba[1] >= 0.5 else 'Negative Sentiment' for proba in prediction]

    # Join the list of sentiments into a single string
    prediction = ', '.join(prediction)
    
    return render_template('tweet_sentiment.html', prediction_text = f'Your tweet has a {prediction}')


if __name__ == '__main__':
    app.run(debug=False)
