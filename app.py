from flask import Flask, render_template, request
from functions import *
import pandas as pd
import spacy
import pickle
import os
import sklearn

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm', disable=['ner'])
bag_tags_lst = pd.read_csv('src/list_tags.csv').list.to_list()


# Initialisation des modèles de prédiction
df_topic_keywords = pd.read_csv('src/df_topic_keywords.csv')
count_vectorizer = pickle.load(open('src/count_vectorizer.sav', 'rb'))
vectorizer = pickle.load(open('src/count_vectorizer.sav', 'rb'))
binarizer = pickle.load(open('src/binarizer.sav', 'rb'))
supervised_model = pickle.load(open('src/supervised_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')




@app.route("/tags", methods=["POST"])
def predict():
    text = request.form['user_question']
    cleaned_text = str(nlp(text))
    sup = supervised_tags(cleaned_text,
                          count_vectorizer,
                          binarizer,
                          supervised_model,
                          treshold=0.11)
   
       
    return render_template('index.html',
                           question=text,
                           sup=sup[0].split(','))

    
if __name__ == "__main__":
        app.run(port=5114)
