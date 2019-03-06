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

from collections import Counter

data_dir = "Data/"
spliter = re.compile(r'[^0-9a-zA-Z]+', re.S)   # Reg that replace all char except number and english
cacheStopWords = nltk.corpus.stopwords.words("english") #dictionary of stop words
counter = Counter()
keys = ['Paragraph','Title','Span','Others']  # content tags that need to be extracted

page_id_list = []
page_content_list = []

# '''
# tokenize the input sentences and update the Word Count data set (Counter)
# '''
# def tokenize(document):
#     document = re.sub(spliter, " ", document)  # replace some special char like -
#     texts_tokenized = [word.lower() for word in nltk.tokenize.word_tokenize(document)] # use nltk to split the english word and ignore some other punctuations
#     return texts_tokenized

'''
merge tag such as Paragraph or Title
'''
def merge_text(page):
    content_str = ''
    for k in keys:
        if len(page[k]) == 0:
            continue
        for sentence in page[k]:
            sentence = re.sub(spliter, " ", sentence)
            if sentence not in cacheStopWords: 
            	content_str += sentence + " "
    return content_str

'''
read all raw data json
loop for ever key(page id)
extract tags and merge them as a string, split by space
'''
def handle_input():
    dirs = os.listdir(data_dir)
    for i in range(0,len(dirs)):
        path = os.path.join(data_dir,dirs[i])
        with open(path,'r',encoding = 'utf-8') as f: 
            data = json.load(f)
            for key in data.keys(): # read a single page
                text = merge_text(data[key])  # merge tag such as Paragraph or Title
                page_id_list.append(key)
                page_content_list.append(text)

'''
construct index using TfidfTransformer
structure (temporal data, not final index) would be :
{
    word1: [(page_id, tfidf),(page_id, tfidf)...],
    word2: []
}
weight_matrix.csv
structure {
		word1 word2 word3
	doc1
	doc2	  tf-idf
	doc3
	.
}
find the document name from  page_id_list
'''
def construct():
    dictionary = {}

    vectorizer = CountVectorizer(analyzer='word', stop_words='english')
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(page_content_list)

    tfidf = transformer.fit_transform(X)
    word = vectorizer.get_feature_names() # already sorted
    weight = tfidf.toarray()

    #write the weight matrix in cvs
    pd_data = pd.DataFrame(weight,index=page_id_list,columns=word)
    pd_data.to_csv('weight_matrix.csv')

    for i in range(len(weight)):
        # print(u"-------Page ",i,u" word list------")
        for j in range(len(word)):
            # print(word[j],weight[i][j])
            if weight[i][j] == 0.0:
                continue
            if word[j] in dictionary.keys():
                dictionary[word[j]].append((page_id_list[i], weight[i][j]))
            else:
                postings = []
                postings.append((page_id_list[i], weight[i][j]))
                dictionary[word[j]] = postings
    return dictionary

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
            postings[pair[0]] = pair[1]
        dictionary[key] = postings

def sort(postings):  
    return sorted(postings,key = lambda pair: pair[1], reverse=True) 
'''
def output(d, outfile="dictionary.json"):
    with open(outfile,'w') as f:
        json.dump(d, f)
'''

def indexer():
    handle_input()  # read all the raw data, and merge content as a string
    dictionary = construct()  # build inverted index
    formalize(dictionary)
    #output(dictionary)
    
if __name__ == '__main__':
    args = sys.argv
    indexer()