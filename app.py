import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import nltk
nltk.data.path.append('/tmp')

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', download_dir='/tmp')

try:
    nltk.data.find('tokenizers/punkt_tab')
except:
    nltk.download('punkt_tab', download_dir='/tmp')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', download_dir='/tmp')
#-------------------------------

def transform_text(text):
    # lower case
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuation

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')
input_sms=st.text_area("Enter the Messages")
if st.button('Predict'):
    #1. preprocess
    transform_sms=transform_text(input_sms)
    # 2. vectorizer
    vector_input = vectorizer.transform([transform_sms])

    #3. predict
    result=model.predict(vector_input)[0]
    # 4.display
    if result==1:
        st.header("spam")
    else:
        st.header('not spam')



