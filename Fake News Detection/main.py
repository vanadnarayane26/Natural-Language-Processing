import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def preprocess(text):
    ps = PorterStemmer()
    headline = re.sub('[^a-zA-Z]'," ",text)
    headline = headline.lower()
    headline = headline.split()
    headline = [ps.stem(word) for word in headline if not word in stopwords.words('english')]
    headline = " ".join(headline)
    return headline

filename ='fake_news_detec_pipeline.pkl'
with open(filename,'rb') as file:
    saved_rf_model = pickle.load(file)
app = FastAPI()

@app.get("/")
async def read_root():
    return {'Fake news detection API!'}

@app.get("/classification")
def predict(text:str):
    text_preprocessed = preprocess(text)
    prediction = saved_rf_model.predict([text_preprocessed])
    if prediction==[1]:
        temp = 'Not Fake'
    else:
        temp = 'Fake'
    return {'Prediction': f'The news is {temp}'}

if __name__ == "__main__":
    uvicorn.run("main:app",host = "127.0.0.1",port=8080,log_level = "info",reload = True)