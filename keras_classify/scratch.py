# keras_classify/scratch.py

import os
import sys
import numpy
import requests
import zipfile

import pandas as pd
import tensorflow 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers



def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def extract_zip(path_to_zip_file):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

sentences = ["John likes ice-cream", "John hates chocolate."]

vectorizer = CountVectorizer(min_df=0.1, lowercase=False)
vectorizer.fit(sentences)
print(vectorizer.vocabulary_) 

print(vectorizer.transform(sentences).toarray())
link =  'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'


#download_url(link, 'data.zip')
#extract_zip('data.zip')


# Extract data and append to a single structure
pathroot = os.getcwd()
filepath_dict = {'yelp':   '\\keras_classify\\data\\yelp_labelled.txt',
                 'amazon': '\\keras_classify\\data\\amazon_cells_labelled.txt',
                 'imdb':   '\\keras_classify\\data\\imdb_labelled.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(pathroot+ filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
df = pd.concat(df_list)

# Extract entries that are labelled as yelp 
# Extract sentences and labels from concatenated data set
print(df.iloc[0])
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values


# Separate the data between training and testing bins  
# The first set the training data to 75% of the sample set, the second does 50%
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

Sentences_train, Sentences_test, y_Train, y_Test = train_test_split(
    sentences, y, test_size=0.5, random_state=1000)

# Vectorize the dataset to produce word map (or vector)
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_Train = vectorizer.transform(Sentences_train)
X_Test = vectorizer.transform(Sentences_test)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
print(X_train)

# Classifier 


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

classifier.fit(X_Train, y_Train)
score2 = classifier.score(X_Test, y_Test)

print("Accuracy: ", score, " Accuracy 2: ", score2)


input_dim = X_train.shape[1] # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

'''

print(f"{n=}, {n+len(ref)=}, {test[n:n+len(ref)]}")
q = test[n:n+len(ref)]
ratio = fuzz.ratio(ref,q)
print(f, q, ratio)
ratios.append(ratio)
n+=1
print(ratio)
'''

