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
import os
import pickle
import scipy.io as io
import csv
import sqlite3 as sql
print(os.getcwd())
cacheStopWords = nltk.corpus.stopwords.words("english") #dictionary of stop words

from collections import Counter

snow = nltk.stem.SnowballStemmer("english")

data_dir = "Data/"
base_dir = os.getcwd()
chunk_dir = "Matrix/"

spliter = re.compile(r'[^0-9a-zA-Z]+', re.S)   # Reg that replace all char except number and english
counter = Counter()
keys = ['Paragraph','Title','Span','Others','Link_Name','url','h1_h2','h3_h5','Strong']  # content tags that need to be extracted
page_id_list = []
page_content_list = []
stemmed_page_list = []
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
# def tokenize():
#     print("function tokenize")
#     '''
#     {
#         word: {
#             page_id: num,
#             page_id: num,
#         }
#     }
#     '''
#     count= {}
#     stemmed_page_list = []
#     for i in range(len(page_content_list)):
#         document = re.sub(spliter, " ", page_content_list[i])  # replace some special char like -
#         texts_tokenized = [snow.stem(word.lower()) for word in nltk.tokenize.word_tokenize(document)] # use nltk to split the english word and ignore some other punctuations
#         for word in texts_tokenized:
#             if word in count.keys():
#                 if page_id_list[i] in count[word].keys():
#                     count[word][page_id_list[i]] += 1
#                 else:
#                     count[word][page_id_list[i]] = 1
#             else:
#                 count[word] = {}
#                 count[word][page_id_list[i]] = 1
#         stemmed_page_list.append(" ".join(texts_tokenized))
#     return count, stemmed_page_list

'''
merge tag such as Paragraph or Title
'''
def merge_text(page, key=""):
    content_str = ''
    texts_tokenized_all = []
    for k in keys:
        if len(page[k]) == 0:
            continue
        for sentence in page[k]:
            sentence = re.sub(spliter, " ", sentence)
            texts_tokenized = [snow.stem(word.lower()) for word in nltk.tokenize.word_tokenize(sentence) if len(word)>1]   
            content_str += sentence + " "
            texts_tokenized_all.extend(texts_tokenized)
    
    stemmed_page_list.append(" ".join(texts_tokenized_all))
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
        print(dirs[i])
        if dirs[i] == 'weight.json':
            continue
        path = os.path.join(data_dir,dirs[i])
        try:
            with open(path,'r',encoding = 'utf-8') as f: 
                data = json.load(f)
                for key in data.keys(): # read a single page
                    text = merge_text(data[key], key=key)  # merge tag such as Paragraph or Title
                    page_id_list.append(key)
                    page_content_list.append(text)
        except:
          pass

'''
construct index using TfidfTransformer
structure (temporal data, not final index) would be :
{
    word1: [(page_id, tfidf),(page_id, tfidf)...],
    word2: []
}
'''
# def construct(count_dict):
#     print("function construct")
#     for word in count_dict.keys():
#         for page_id in count_dict[word].keys():
#             if word in dictionary.keys():
#                 dictionary[word].append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))   ))
#             else:
#                 postings = []
#                 postings.append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))  ))
#                 dictionary[word] = postings
#     return dictionary

def insert_db(tuples):
    con = sql.connect('test.db')
    cur = con.cursor()
    cur.execute("DROP TABLE matrix")
    cur.execute("CREATE TABLE IF NOT EXISTS matrix (id INTEGER PRIMARY KEY, array BLOB)")
    cur.executemany("INSERT INTO matrix VALUES (?,?)", tuples )
    con.commit()
    # cur.execute("SELECT * FROM matrix")

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

def Matrix_Generator(stemmed_page_list, page_id_list, chunk = 100):
    print("Start generate Matrix")
    with open(os.path.join(base_dir,'weight.json'),'r',encoding = 'utf-8') as f: 
        weight = json.load(f)
    dictionary = {}
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(stemmed_page_list))
    word = vectorizer.get_feature_names() # already sorted
    Matrix = tfidf.toarray()
    print(len(Matrix),len(Matrix[0]))
    # pd_data = pd.DataFrame(Matrix,index=page_id_list,columns=word)
    # pd_data.to_csv('weight_matrix.csv')
    insert_items = []
    for i in range(0,4000):
        print("Exporting:",str(i))
        if chunk*(i+1)>len(Matrix):
            insert_items.append( (i, json.dumps(Matrix[i*chunk:len(Matrix),:].tolist())) )
        else:
            insert_items.append( (i, json.dumps(Matrix[i*chunk:chunk*(i+1),:].tolist())) )
    insert_db(insert_items)
    return
    for i in range(0,len(Matrix[0])):
        for j in range(len(Matrix)):
            if(Matrix[j][i]!=0):
                if word[i] in weight.keys():
                    # print(i,j)
                    if page_id_list[j] in weight[word[i]].keys():
                        log_weight = math.log(weight[word[i]][page_id_list[j]],10)
                        Matrix[j][i] *= log_weight
    print("Weighted")
    for i in range(0,4000):
        print("Exporting:",str(i))
        if chunk*(i+1)>len(Matrix):
            with open(os.path.join(chunk_dir,'weight_matrix_'+str(i)+'.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerows(Matrix[i*chunk:len(Matrix),:])
            break
            # pd_data = pd.DataFrame(Matrix[i*chunk:len(Matrix),:])
            # pd_data.to_csv('weight_matrix_'+str(i)+'.csv')
            # pickle.dump(Matrix[i*chunk:len(Matrix),:], open('weight_matrix_'+str(i)+'.csv', "w"))
        else:
            with open(os.path.join(chunk_dir,'weight_matrix_'+str(i)+'.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerows(Matrix[i*chunk:chunk*(i+1),:])
            # pd_data = pd.DataFrame(Matrix[i*chunk:chunk*(i+1),:])
            # pd_data.to_csv('weight_matrix_'+str(i)+'.csv')        
            # pickle.dump(Matrix[i*chunk:chunk*(i+1),:], open('weight_matrix_'+str(i)+'.csv', "w"))
    # #write the Matrix in cvs
    # pd_data = pd.DataFrame(Matrix,index=page_id_list,columns=word)
    # pd_data.to_csv('weight_matrix.csv')

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
    # handle_input()  # read all the raw data, and merge content as a string
    # temp = {}
    # temp["list"] = stemmed_page_list
    # temp["ids"] = page_id_list
    # output(temp, outfile="stem_list.json")
    with open(os.path.join(base_dir,"stem_list.json"),'r',encoding = 'utf-8') as f:
        j = json.load(f)
        stemmed_page_list = j["list"]
        page_id_list = j["ids"]
    Matrix_Generator(stemmed_page_list, page_id_list)
    #formalize(dictionary)
    #output(dictionary)
    
    
if __name__ == '__main__':
    args = sys.argv
    indexer()
