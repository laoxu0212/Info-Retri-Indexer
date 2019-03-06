#MARCH 5th, Xiangyu Version
import json
import sys
import os
import re
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


raw_dir = 'WEBPAGES_RAW'
global dictionary
global page_ids
global weight_matrix

with open('dictionary.json','r',encoding = 'utf-8') as f: 
    dictionary = json.load(f)
with open(os.path.join(raw_dir,'bookkeeping.json'),'r',encoding = 'utf-8') as f: 
    page_ids = json.load(f)
try:
    weight_matrix = pd.read_csv('weight_matrix.csv')
except FileNotFoundError:
    print('file not found!!')
else:
    term_list = list(weight_matrix.columns.values)#terms list
    document_ids = list(weight_matrix.iloc[:,0].values)#document IDs list

cacheStopWords = nltk.corpus.stopwords.words("english")

#Queries
def build_queryVec(query_list):
    #calculate idf for each word in query
    totalDoc = 37497
    idf_list = [0]*(len(term_list)-1)
    for i in range(len(query_list)):
        if dictionary.__contains__(query_list[i]):
            idf_list[term_list.index(query_list[i])-1] = totalDoc/len(dictionary[query_list[i]])

    #calculate tf for each word
    query_vec = [0]*(len(term_list)-1)
    for word in query_list:
        if word in term_list:
            query_vec[term_list.index(word)-1] += 1
    #generate query vector and nomalize
    for j in range(len(query_vec)):
        query_vec[j] = query_vec[j]*idf_list[j]
    query_vec = np.array(query_vec)
    #Do nomalization here
    np.seterr(divide='ignore',invalid='ignore')
    query_vec = query_vec / np.linalg.norm(query_vec)
    return query_vec


def search(query):

    if len(query)==0:
        print("No query！")
        return
    #pre-process the query and put each word in list:query_list
    cleaner = re.compile(r'[^0-9a-zA-Z]+', re.S)
    query_list = [item.lower() for item in query]
    for item in query_list:
        item = re.sub(cleaner, ' ', item)
        item = nltk.tokenize.word_tokenize(item)
    print(query_list)

    query_vec = build_queryVec(query_list).reshape(1,-1)

    flag = False
    result = {}
    document_list = dictionary.keys()
    for word in query_list:
        if dictionary.__contains__(word):
            flag = True
            postings = dictionary[word]
            #print(postings)
            for document in postings.keys():
                if document in document_ids:
                    index = document_ids.index(document)
                    document_vec = np.float32(weight_matrix.iloc[index,1:].values).reshape(1,-1)
                    '''
                    print(document)
                    print(query_vec.shape)
                    print(document_vec.shape)
                    print(query_vec)
                    print(document_vec)
                    '''
                    score = cosine_similarity(query_vec,document_vec)[0][0]
                    result[document] = score


    if flag==False:
        print("not found!")
        return
    result = sorted(result.items(),key=lambda d:d[1],reverse=True)
    for res in result:
        print(page_ids[res[0]])


if __name__ == '__main__':
    query = sys.argv[1:]
    search(query)