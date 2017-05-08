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

# Cleaning the Tweets
import preprocessor as p
def clean_tweets(a):
    clean = p.clean(a)
    return clean
dataset['Clean Tweets'] = dataset['Tweet'].apply(lambda x : clean_tweets(x))
