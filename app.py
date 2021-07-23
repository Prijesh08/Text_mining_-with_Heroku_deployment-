import pandas as pd

import numpy as np

import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from flask import Flask,render_template,request


data = pd.read_csv('data.csv')


split = data[['reviews', 'senti']]
test = split.sample(frac=0.2,random_state = 200)
train = split.drop(test.index)
print(test.shape)
print(train.shape)

en_model=spacy.load('en_core_web_sm')
sw_spacy=en_model.Defaults.stop_words

count_vect = CountVectorizer(min_df=2 ,stop_words=sw_spacy , ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()

all_features = count_vect.fit_transform(data['reviews'].values.astype('U'))
# print(all_features.shape)
count_vect.vocabulary_

#for training
X_train_counts = count_vect.fit_transform(train["reviews"].values.astype('U'))        
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#for testing
X_new_counts = count_vect.transform(test["reviews"].values.astype('U'))
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

sm=SMOTE(random_state=444)
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, train.senti)


model=MLPClassifier(hidden_layer_sizes=(5,5))
model.fit(X_train_res,y_train_res)



app= Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    input_value=str(request.form['message'])
    # print("*********** "+input_value)
    
    if len(input_value) > 0:
        tdm = count_vect.transform([input_value])

        
        output=model.predict(tdm)

        if output==1:
            res_val= "POSITIVE"
            return render_template('home.html',prediction_text= f'Your Review:"{input_value}" is POSITIVE')
        else:
            res_val='NEGATIVE'
            return render_template('home.html',prediction_text= f'Your Review:"{input_value}" is NEGATIVE')
    else:
        # res_val = 'enter'
        return render_template('home.html',prediction_text= 'Enter the reviews')

if __name__ == "__main__":
    app.run(debug=True)