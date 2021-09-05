import nltk
import pickle
import os
import json
import glob
import re
import sys


path = os.getcwd()
os.chdir(path)

def final(string , l):
  res = string + ": "
  for i in l:
    res = res + "<" + i[0] + "," + str(i[1]) + ">,"
  res = res + ";"
  return res

example_dict={}
pickle_in = open(path+"/ECTInvertedIndex.pickle","rb")
example_dict = pickle.load(pickle_in)

#print(example_dict)
queries_file = sys.argv[1]
queries = open(queries_file).readlines()
queries = [query for query in queries]

for i in queries:
	print(i)

f = open("RESULTS1_20CS60R56.txt", "w")

for query in queries:
	position = query.find('*') 
	if position>=0 :
		regex = "^" + query[:position] + "." + query[position:] + "$"
		for word in example_dict.keys():
			match = re.search(regex , word)
			if match:
				#print(example_dict[word])
				res = final(word , example_dict[word])
				#print(res)
				f.write(res)

	else :
		for word in example_dict.keys():
			if word == query:
				#print(example_dict[word])
				res = final(word , example_dict[word])
				#print(res)
				f.write(res)

f.close()
