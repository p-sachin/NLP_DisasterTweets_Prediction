# Importing the libraries
import unicodedata
import string
import joblib
import re
import spacy

# Load the spacy library for text cleaning
nlp = spacy.load('en_core_web_sm')

# Loading the saved model
rf_clf = joblib.load('models/rf_clf.pkl')

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427


def unicode_to_ascii(s):
    all_letters = string.ascii_letters + " .,;'-"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Remove Stop Words


def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])


def remove_duplicates(text):
    if len(text) < 2:
        return text
    if text[0] != text[1]:
        return text[0]+remove_duplicates(text[1:])
    return remove_duplicates(text[1:])


def clean_text(text):
    # Text to lowercase
    text = text.lower()
    # Remove URL from text
    text = re.sub(r"http\S+", "", text)
    # Remove Numbers from text
    text = re.sub(r'\d+', '', text)
    # Convert the unicode string to plain ASCII
    text = unicode_to_ascii(text)
    # Remove Punctuations
    text = re.sub(r'[^\w\s]', '', text)
    #text = remove_punct(text)
    # Remove StopWords
    text = remove_stopwords(text)
    # Remove empty spaces
    text = text.strip()
    # \s+ to match all whitespaces
    # replace them using single space " "
    text = re.sub(r"\s+", " ", text)

    #text = remove_duplicates(text)

    return text


# Prediction
guess = False
while not guess:
    my_tweet = input("Enter your tweet: ")
    my_tweet = clean_text(my_tweet)
    prediction = rf_clf.predict([my_tweet])
    if prediction[0] == 0:
        print("Prediction: Not a Disaster Tweet")
        guess = True
    else:
        print("Prediction: It is a Disaster Tweet")
        guess = True
