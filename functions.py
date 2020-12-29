import numpy as np
import pandas as pd
import pickle 
import re, spacy, nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
import sklearn

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from bs4 import BeautifulSoup
from spacy.tokens import Doc
from spacy.language import Language

from sklearn.multiclass import OneVsRestClassifier

stop_words = set(stopwords.words("english"))
token = ToktokTokenizer()
punct = punctuation



def clean_text(text):
        ''' Lowering text and removing undesirable marks
        Parameter:
        text: document to be cleaned    
        '''

        text = text.lower()
        text = re.sub(r"\'\n", " ", text)
        text = re.sub(r"\'\xa0", " ", text)
        text = re.sub('\s+', ' ', text)  # matches all whitespace characters
        text = text.strip(' ')
        return text
	
def strip_list_noempty(mylist):

    newlist = (item.strip() if hasattr(item, 'strip')
               else item for item in mylist)

    return [item for item in newlist if item != '']

def clean_punct(text, top_tags):
    ''' Remove all the punctuation from text, unless it's part of an important 
    tag (ex: c++, c#, etc)
    Parameter:
    text: document to remove punctuation from it
    '''

    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    top_tags = top_tags

    for w in words:
        if w in top_tags:
            punctuation_filtered.append(w)
        else:
            w = re.sub('^[0-9]*', " ", w)
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))
	
def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus
    Parameter:
    text: document to remove stop words from it
    '''

    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))

def lemmatization(texts, allowed_postags, top_tags,stop_words=stop_words):
        ''' It keeps the lemma of the words (lemma is the uninflected form of a word),
        and deletes the underired POS tags
        Parameters:
        texts (list): text to lemmatize
        allowed_postags (list): list of allowed postags, like NOUN, ADL, VERB, ADV
        '''

        lemma = wordnet.WordNetLemmatizer()
        doc = nlp(texts)
        texts_out = []
        top_tags = top_tags
		
        for token in doc:

            if str(token) in top_tags:
                texts_out.append(str(token))

            elif token.pos_ in allowed_postags:

                if token.lemma_ not in ['-PRON-']:
                    texts_out.append(token.lemma_)

                else:
                    texts_out.append('')

        texts_out = ' '.join(texts_out)

        return texts_out
    
def transform_tuple(tup):
    i = 0
    for sub in tup:
        tup[i] = ','.join(sub)
        i += 1
    return tup
    
def supervised_tags(cleaned_text, vectorizer, binarizer,
                    supervised_model, treshold=0.11):
    tfidf_cleaned_text = vectorizer.transform([cleaned_text])
    pred = supervised_model.predict_proba(tfidf_cleaned_text)
    pred = pd.DataFrame(pred).applymap(lambda x: 1 if x > treshold else 0)
    pred = pred.to_numpy()
    return transform_tuple(binarizer.inverse_transform(pred))
