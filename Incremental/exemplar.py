import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import json, re, nltk, string
from nltk.corpus import wordnet
nltk.download('punkt')

def split_new_data(new_class_data):
    data = new_class_data
    train, test = train_test_split(data, test_size=0.000000001, random_state=42)
    return train, test



def exemplar(basefile, newfile, sp):
    base_data = pd.read_csv(basefile, encoding = "latin-1")
    selected = ['assigned_to', 'description']
    #selected = ['assigned_to', 'description', 'summary']
    non_selected = list(set(base_data.columns) - set(selected))

    base_data = base_data.drop(non_selected, axis=1) # Drop non selected columns
    base_data = base_data.dropna(axis=0, how='any', subset=selected) # Drop null rows
    base_data = base_data.reindex(np.random.permutation(base_data.index))

    samples = int(len(base_data.index)*sp)
    #print(samples)

    a=random.sample(range(1, len(base_data.index)), samples)
    data=base_data.loc[a]

    new_data= pd.read_csv(newfile, encoding = "latin-1")
    selected = ['assigned_to', 'description']
    #selected = ['assigned_to', 'description', 'summary']
    non_selected = list(set(new_data.columns) - set(selected))

    new_data = new_data.drop(non_selected, axis=1) # Drop non selected columns
    new_data = new_data.dropna(axis=0, how='any', subset=selected) # Drop null rows
    new_data = new_data.reindex(np.random.permutation(new_data.index))
    train, test = split_new_data(new_data)

    example_data = train.append(data, ignore_index=True)
    example_data.to_csv('./../data/exampledata.csv',index=False)
    #data_t=pd.read_csv('./../data/inc_test_data_2.csv', encoding = "latin-1")
    #test_data = test.append(data_t, ignore_index=True)
    test.to_csv('./../data/inc_test_data.csv', index=False)  #test_data.to_csv('./../data/inc_test_data_3.csv', index=False)
    #print(example_data)
    return example_data, test, new_data


def data_preprocess(df):
    all_data = []
    all_owner = []
    for row in range(len(df)):
        item=df.iloc[row,:]
        #1. Remove \r
        text = item['description'].replace('\r', ' ')
        #2. Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        #3. Remove Stack Trace
        start_loc = text.find("Stack trace:")
        text = text[:start_loc]
        #4. Remove hex code
        text = re.sub(r'(\w+)0x\w+', '', text)
        #5. Change to lower case
        text = text.lower()
        #6. Tokenize
        text = nltk.word_tokenize(text)
        #7. Strip punctuation marks
        text = [word.strip(string.punctuation) for word in text]
        #8. Join the lists
        all_data.append(text)
        all_owner.append(item['assigned_to'])

    all_data=[' '.join([j for j in i if len(j)>1]) for i in all_data]

    df=pd.DataFrame(list((all_data,all_owner)),index=['description','assigned_to']).T
    df.head()
    #df["sentences"] = df["sentences"].replace(np.nan, 'none', regex=True)
    #df["labels"] = df["labels"].replace(np.nan, 'none', regex=True)
    #sentences = df.sentences.values
    #labels = df.labels.values
    return df


#print(exemplar('./data1.csv', './data12.csv'))