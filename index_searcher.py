import json
import sys
import os

raw_dir = 'WEBPAGES_RAW'
global dictionary
global page_ids

with open('dictionary.json','r',encoding = 'utf-8') as f: 
    dictionary = json.load(f)
with open(os.path.join(raw_dir,'bookkeeping.json'),'r',encoding = 'utf-8') as f: 
    page_ids = json.load(f)

def search(word):
    if word not in dictionary.keys():
        print("not found!")
        return
    postings = dictionary[word]
    for page_id in postings.keys():
        print(page_ids[page_id])

if __name__ == '__main__':
    args = sys.argv
    search(args[1])