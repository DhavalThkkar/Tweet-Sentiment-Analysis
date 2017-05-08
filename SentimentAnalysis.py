#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:47:49 2017

@author: thakkar_
"""
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('sentiment.tsv', delimiter = '\t', quoting=3)

# Converting the 'Sentiment' column from 'neg','pos' to 0 ,1
def strToInt(a):
    if a == 'neg':
        return 0
    elif a == 'pos':
        return 1
dataset['Sentiment'] = dataset['Sentiment'].apply(lambda x : strToInt(x))

# Cleaning the Tweets
import preprocessor as p
def clean_tweets(a):
    clean = p.clean(a)
    return clean
dataset['Clean Tweets'] = dataset['Tweet'].apply(lambda x : clean_tweets(x))

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 2001):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Clean Tweets'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values
y = np.reshape(y,(2001,1))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 2000))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'tanh'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_test_bool = (y_test >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test_bool,y_pred))
