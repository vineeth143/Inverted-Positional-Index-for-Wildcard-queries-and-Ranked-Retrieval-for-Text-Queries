import nltk
import pickle
import os
import json
import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import BracketParseCorpusReader
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
stop_words = stop_words.union(",","(",")","[","]","{","}","#","@","!",":",";",".","?","-",":")

Index={}

tokens = []
lemmatizer = WordNetLemmatizer()

path = os.getcwd()
os.chdir(path)

for file in glob.glob(path+"/ECTText/*.txt"):
	o=0
	document = open(file,encoding='utf8').read()
	file_name=file.split('/')
	doc_ID=file_name[-1]
	print(doc_ID)
	tokens = word_tokenize(document)
	for token in tokens:
		optimized_word = lemmatizer.lemmatize(token)
		position =document[o:].find(token) + o
		o=o+len(token)
		while document[o]==' ':
			o=o+1

		if optimized_word not in stop_words and not optimized_word.isdigit():
			if optimized_word not in Index.keys():
				Index.update({optimized_word : []})
			Index[optimized_word].append( (doc_ID , position) )

pickle_out = open('ECTInvertedIndex.pickle', 'wb')
pickle.dump(Index, pickle_out)
pickle_out.close()
print(Index)