import nltk
import re
import string
import argparse
import sys
import os
import json
import pandas as pd
import heapq
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
cacheStopWords = nltk.corpus.stopwords.words("english") #dictionary of stop words

from collections import Counter

snow = nltk.stem.SnowballStemmer("english")

data_dir = "Data/"
spliter = re.compile(r'[^0-9a-zA-Z]+', re.S)   # Reg that replace all char except number and english
counter = Counter()
keys = ['Paragraph','Title','Span','Others','Link_Name','url','h1_h2','h3_h5','Strong']  # content tags that need to be extracted
page_id_list = []
page_content_list = []
tag_weight = {}

tag_value = {   'Title': 10,
                'url': 7,
                'Link_Name': 6,
                'h1_h2': 5,
                'Strong': 2,
                'Paragraph': 1,
                'Span': 1,
                'h3_h5': 1,
                'Others': 1}


'''
tokenize the input sentences and update the Word Count data set (Counter)
'''
def tokenize():
    print("function tokenize")
    '''
    {
        word: {
            page_id: num,
            page_id: num,
        }
    }
    '''
    count= {}
    stemmed_page_list = []
    for i in range(len(page_content_list)):
        document = re.sub(spliter, " ", page_content_list[i])  # replace some special char like -
        texts_tokenized = [snow.stem(word.lower()) for word in nltk.tokenize.word_tokenize(document)] # use nltk to split the english word and ignore some other punctuations
        for word in texts_tokenized:
            if word in count.keys():
                if page_id_list[i] in count[word].keys():
                    count[word][page_id_list[i]] += 1
                else:
                    count[word][page_id_list[i]] = 1
            else:
                count[word] = {}
                count[word][page_id_list[i]] = 1
        stemmed_page_list.append(" ".join(texts_tokenized))
    return count, stemmed_page_list

'''
merge tag such as Paragraph or Title
'''
def merge_text(page, key=""):
    print("Let's merge")
    content_str = ''
    for k in keys:
        if len(page[k]) == 0:
            continue
        for sentence in page[k]:
            #print(k)
            sentence = re.sub(spliter, " ", sentence)
            texts_tokenized = [snow.stem(word.lower()) for word in nltk.tokenize.word_tokenize(sentence)]
            for word in texts_tokenized:
                if word not in tag_weight.keys():
                    tag_weight[word] = dict()
                if key in tag_weight[word].keys():
                    tag_weight[word][key] = tag_weight[word][key] + tag_value[k]
                else:
                    tag_weight[word][key] = tag_value[k]         
            content_str += sentence + " "
    return content_str

'''
read all raw data json
loop for ever key(page id)
extract tags and merge them as a string, split by space
'''
def handle_input():
    print('handle_input')
    dirs = os.listdir(data_dir)
    for i in range(0,len(dirs)):
        path = os.path.join(data_dir,dirs[i])
        with open(path,'r',encoding = 'utf-8') as f: 
            data = json.load(f)
            for key in data.keys(): # read a single page
                text = merge_text(data[key], key=key)  # merge tag such as Paragraph or Title
                page_id_list.append(key)
                page_content_list.append(text)

'''
construct index using TfidfTransformer
structure (temporal data, not final index) would be :
{
    word1: [(page_id, tfidf),(page_id, tfidf)...],
    word2: []
}
'''
def construct(count_dict):
    print("function construct")
    for word in count_dict.keys():
        for page_id in count_dict[word].keys():
            if word in dictionary.keys():
                dictionary[word].append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))   ))
            else:
                postings = []
                postings.append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))  ))
                dictionary[word] = postings
    return dictionary

'''
weight_matrix.csv
structure {
        word1 word2 word3
    doc1
    doc2      tf-idf
    doc3
    .
}
find the document name from  page_id_list
'''

def Matrix_Generator(stemmed_page_list):
    print("Start generate Matrix")
    with open('weight.json','r',encoding = 'utf-8') as f: 
        weight = json.load(f)
    dictionary = {}
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(stemmed_page_list))
    word = vectorizer.get_feature_names() # already sorted
    Matrix = tfidf.toarray()
    for i in range(1,len(Matrix[0])):
        for j in range(len(Matrix)):
            if(Matrix[j][i]!=0):
                if word[i] in weight.keys():
                    if page_id_list[j] in weight[word[i]].keys():
                        log_weight = math.log(weight[word[i]][page_id_list[j]],10)
                        Matrix[j][i] *= log_weight
    #write the Matrix in cvs
    pd_data = pd.DataFrame(Matrix,index=page_id_list,columns=word)
    pd_data.to_csv('weight_matrix.csv')

'''
transform the dict to json format.
sort postings based on tfidf.
structure (final index) would be :
{
    word1: {
        page_id: tfidf,
        page_id: tfidf
    },
    word2: []
}
'''
def formalize(dictionary):
    for key in dictionary.keys():
        postings = {}
        sort(dictionary[key])
        for pair in dictionary[key]:  # transfor from list to dict
            postings[pair[0]] = round(pair[1],4)
        dictionary[key] = postings

def sort(postings):  
    return sorted(postings,key = lambda pair: pair[1], reverse=True) 

def calculate_tfidf(tf,df):
    if tf==0 or df==0:
        return 0
    else:
        return (tf) * (np.log10(  (len(page_id_list)+1)/  (df+1) ) +1)

def output(d, outfile="dictionary.json"):
    with open(outfile,'w') as f:
        json.dump(d, f)

def indexer():
    handle_input()  # read all the raw data, and merge content as a string
    count, stemmed_page_list = tokenize()
    #dictionary = construct(count)  # build inverted index
    Matrix_Generator(stemmed_page_list)
    #formalize(dictionary)
    #output(dictionary)
    output(tag_weight, outfile="weight.json")
    
if __name__ == '__main__':
    args = sys.argv
    indexer()
