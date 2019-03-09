
import json
import heapq
import sys
import os
import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import sqlite3 as sql


raw_dir = 'WEBPAGES_RAW'
global dictionary
global page_ids
global weight_matrix

with open('dictionary.json','r',encoding = 'utf-8') as f: 
    dictionary = json.load(f)
with open(os.path.join(raw_dir,'bookkeeping.json'),'r',encoding = 'utf-8') as f: 
    page_ids = json.load(f)
with open('stem_list','r',encoding = 'utf-8') as f: 
    Mapping = json.load(f)

'''
Mapping = {list:[doc1, doc2,...],ids : [id1, id2 ...]}
'''
try:
    con = sql.connect('test.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM matrix")
    d = cur.fetchall()
    Matrix = []
    for i in range(len(d)):
        for _list in eval(d[i][1]):
            Matrix.append(_list)
except FileNotFoundError:
    print('file not found!!')
else:
    term_list = Mapping['vocabulary']#terms list
    document_ids = Mapping['ids']#document IDs list

cacheStopWords = nltk.corpus.stopwords.words("english")

#Queries
def build_queryVec(query_list):
    #calculate idf for each word in query
    totalDoc = 37497
    idf_list = [0]*(len(term_list)-1)
    for i in range(len(query_list)):
        if dictionary.__contains__(query_list[i]) and query_list[i] in term_list:
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
    query_vec = query_vec / np.linalg.norm(query_vec)
    return query_vec


def search(query):

    if(len(query)==0):
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
    result = []
    posting_set = set()
    
    for word in query_list:
        if dictionary.__contains__(word):
            flag = True
            postings = dictionary[word]
            for document in postings.keys():
                posting_set.add(document)

    if flag==False:
    	print("not found!")
    	return
    for document in posting_set:
        if document in document_ids:
            index = document_ids.index(document)
            document_vec = np.float32(Matrix[index,1:].values).reshape(1,-1)
            '''
            print(document)
            print(query_vec.shape)
            print(document_vec.shape)
            print(query_vec)
            print(document_vec)
            '''
            score = cosine_similarity(query_vec,document_vec)[0][0]
            insert(result,(score,document))

    #The situation that score is toooo low
    '''
    if(result[result[]]<0.0001):
        print("Did not match any documents.")
        return
    '''
    #result = [(score,url)]
    result_list = []
    while len(result)!=0:
    	res = heapq.heappop(result)
    	result_list.append(page_ids[res[1]])

    result_list.reverse()
    for res in result_list:
    	print(res)
#To maintain a max-heap with size 100
def insert(a, val):
	if(len(a) >= 100):
		if(val > a[0]):
			heapq.heappop(a)
			heapq.heappush(a, val)
	else:
		heapq.heappush(a, val)





if __name__ == '__main__':
    query = sys.argv[1:]
    search(query)
