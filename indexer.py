import nltk
import re
import string
import argparse
import sys
import os
import json
import heapq
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from collections import Counter

data_dir = "Data/"
spliter = re.compile(r'[^0-9a-zA-Z]+', re.S)   # Reg that replace all char except number and english
counter = Counter()
keys = ['Paragraph','Title','Span','Others']  # content tags that need to be extracted

page_id_list = []
page_content_list = []

'''
tokenize the input sentences and update the Word Count data set (Counter)
'''
def tokenize():
    '''
    {
        word: {
            page_id: num,
            page_id: num,
        }
    }
    '''
    count = {}
    for i in range(len(page_content_list)):
        document = re.sub(spliter, " ", page_content_list[i])  # replace some special char like -
        texts_tokenized = [word.lower() for word in nltk.tokenize.word_tokenize(document)] # use nltk to split the english word and ignore some other punctuations
        for word in texts_tokenized:
            if word in count.keys():
                if page_id_list[i] in count[word].keys():
                    count[word][page_id_list[i]] += 1
                else:
                    count[word][page_id_list[i]] = 1
            else:
                count[word] = {}
                count[word][page_id_list[i]] = 1
    return count

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
'''
def construct(count_dict):
    dictionary = {}
    for word in count_dict.keys():
        for page_id in count_dict[word].keys():
            if word in dictionary.keys():
                dictionary[word].append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))   ))
            else:
                postings = []
                postings.append((page_id, calculate_tfidf(count_dict[word][page_id], len(count_dict[word].keys()))  ))
                dictionary[word] = postings
    return dictionary
# def construct():
#     dictionary = {}
#     vectorizer = CountVectorizer()
#     transformer = TfidfTransformer()
#     tfidf = transformer.fit_transform(vectorizer.fit_transform(page_content_list))
#     word = vectorizer.get_feature_names() # already sorted
#     weight = tfidf.toarray()
#     for i in range(len(weight)):
#         # print(u"-------Page ",i,u" word list------")
#         for j in range(len(word)):
#             # print(word[j],weight[i][j])
#             if weight[i][j] == 0.0:
#                 continue
#             if word[j] in dictionary.keys():
#                 dictionary[word[j]].append((page_id_list[i], weight[i][j]))
#             else:
#                 postings = []
#                 postings.append((page_id_list[i], weight[i][j]))
#                 dictionary[word[j]] = postings
#     return dictionary

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
    dictionary = construct(tokenize())  # build inverted index
    formalize(dictionary)
    output(dictionary)
    
if __name__ == '__main__':
    args = sys.argv
    indexer()