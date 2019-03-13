

import json
import heapq
import math
import os
import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


raw_dir = 'WEBPAGES_RAW'
global dictionary
global page_ids
global weight_matrix

with open('stem_dict.json','r',encoding = 'utf-8') as f: 
    dictionary = json.load(f)
with open(os.path.join(raw_dir,'bookkeeping.json'),'r',encoding = 'utf-8') as f: 
    page_ids = json.load(f)
with open('stem_list.json','r',encoding = 'utf-8') as f: 
    Mapping = json.load(f)

'''
Mapping = {list:[doc1_content, doc2_content,...],ids : [id1, id2 ...]}
'''

content_list = Mapping['list']#content list
document_ids = Mapping['ids']#document IDs list
cacheStopWords = nltk.corpus.stopwords.words("english")

def Matrix_Generator():
    print("Start generate Matrix")
    # with open('weight.json','r',encoding = 'utf-8') as f: 
    #     weight = json.load(f)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(Mapping['list']))
    term_list = vectorizer.get_feature_names() # already sorted

    Matrix = tfidf.toarray()
    # for i in range(len(Matrix[0])):
    #     for j in range(len(Matrix)):
    #         if(Matrix[j][i]==0 and term_list[i] in weight.keys()):
    #             log_weight = math.log(weight[term_list[i]][document_ids[j]],10)
    #             Matrix[j][i] *= log_weight
                
    # print('weighted')
    return Matrix,term_list

#Queries
def build_queryVec(query_list,term_list):
    #calculate idf for each word in query
    print(len(term_list))
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


def search(query,Matrix,term_list):

    if(len(query)==0):
        print("No query!!!")
        return
    #pre-process the query and put each word in list:query_list
    query = query.lower()
    cleaner = re.compile(r'[^0-9a-zA-Z]+', re.S)
    query = re.sub(cleaner, ' ', query)
    q = nltk.tokenize.word_tokenize(query)
    snow = nltk.stem.SnowballStemmer("english") #stemmer
    query_list = [snow.stem(word) for word in q if word not in cacheStopWords and word in term_list]
    print(query_list)
    if(len(query_list)==0):
        print("No matching result")
        return

    query_vec = build_queryVec(query_list,term_list).reshape(1,-1)

    result = []
    posting_set = set()
    
    for word in query_list:
        postings = dictionary[word]
        for document in postings.keys():
            posting_set.add(document)


    for document in posting_set:
        if document in document_ids:
            index = document_ids.index(document)
            document_vec = Matrix[index,1:].reshape(1,-1)
            '''
            print(document)
            print(query_vec.shape)
            print(document_vec.shape)
            print(query_vec)
            print(document_vec)
            '''
            score = cosine_similarity(query_vec,document_vec)[0][0]
            insert(result,(score,document))

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
    Matrix, term_list = Matrix_Generator()
    while(True):
        query = input("What can I do for you: ")
        search(query,Matrix,term_list)
