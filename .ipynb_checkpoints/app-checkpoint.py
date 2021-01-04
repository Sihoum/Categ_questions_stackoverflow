from flask import Flask, render_template, request
from functions import *
import pandas as pd
import spacy
import pickle
import os
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer



app = Flask(__name__)

nlp = spacy.load('en_core_web_sm', disable=['ner'])
bag_tags_lst = pd.read_csv('src/list_tags.csv').list.to_list()


# Initialisation des modèles de prédiction
df_topic_keywords = pd.read_csv('src/df_topic_keywords.csv')
count_vectorizer = pickle.load(open('src/count_vectorizer.sav', 'rb'))
vectorizer = pickle.load(open('src/count_vectorizer.sav', 'rb'))
binarizer = pickle.load(open('src/binarizer.sav', 'rb'))
supervised_model = pickle.load(open('src/supervised_model.sav', 'rb'))
top_tags = pd.read_csv('src/top_tags.csv').list.to_list()



@app.route('/')
def home():
    return render_template('index.html')




@app.route("/tags", methods=["POST"])
def predict():
    text = request.form['user_question']
    cl_txt = clean_text(text)
    cl_txt = clean_punct(cl_txt, top_tags)
    cl_txt = stopWordsRemove(cl_txt)
    cl_txt = lemmatization(cl_txt, ['NOUN', 'ADV'], top_tags,stop_words=stop_words)
    sup = supervised_tags(cl_txt,
                          count_vectorizer,
                          binarizer,
                          supervised_model,
                          treshold=0.11)
   
       
    return render_template('index.html',
                           question=text,
                           sup=sup[0].split(','))

    
if __name__ == "__main__":
        app.run(port=5117)
