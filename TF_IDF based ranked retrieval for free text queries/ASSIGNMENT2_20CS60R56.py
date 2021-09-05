import nltk
import pickle
import numpy as np
import os
import glob
from collections import defaultdict
from bs4 import BeautifulSoup
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import BracketParseCorpusReader
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
import math
import re
import sys
import operator
import copy

path=os.getcwd()
#print(str(path))
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
stop_words = stop_words.union(",","(",")","[","]","{","}","#","@","!",":",";",".","?" , "-" , ":" , "%")

os.chdir(r'../Dataset')

ExampleDict=dict()
static_quality_score=dict()
pickle1 = open("StaticQualityScore.pkl", 'rb')
static_quality_score = pickle.load(pickle1)
ExampleDict=copy.deepcopy(static_quality_score)
count=1
g=0
for i in ExampleDict:
    g=g+1



FilenamesList=list()
pickle2 = open("Leaders.pkl", 'rb')
leaders = pickle.load(pickle2)    
for i in range(len(leaders)):
    leaders[i]=str(leaders[i])+'.html'
    FilenamesList.append(leaders[i])
FilenamesList.sort()

def score(file,Q):
    number=0.0
    ListNumbers=[]
    d1=0.0
    d2=0.0
    Listd1=[]
    Listd2=[]
    #print(Q)
    for term in Q:
        #print(IDF[term],tf_idf[(term,file)])
        try:
            VQt = IDF[term]
            #print(V_Q)
        except:
            VQt=0
        try:
            Vdt = tf_idf[(term,file)]
            #print(V_d)
        except:
            Vdt = 0
        ListNumbers.append(VQt*Vdt)
        number=number+VQt*Vdt
        Listd1.append(VQt**2)
        d1=d1+(VQt)**2
        Listd2.append(Vdt**2)
        d2=d2+(Vdt)**2
        d=(d1**0.5)*(d2**0.5)
    #print(round((number/d),4))
    ListNumbers_sum=sum(ListNumbers)
    Listd1_sum=sum(Listd1)
    Listd2_sum=sum(Listd2)
    return (round((number/d),4))

def manipulate_file(file):
    F = open(file, encoding = 'utf8')
    soup = BeautifulSoup(F,'html5lib')
    text = soup.get_text()
    UpperText=soup.get_text()
    UppperText=UpperText.upper()
    text = text.lower()
    UpperTokens=word_tokenize(UpperText)
    tokens = word_tokenize(text)
    #words=[i for i in tokens if i not in stop_words]
    words=[]
    for i in tokens:
        if i not in stop_words:
            words.append(i)
    wordnet_lemmatizer = WordNetLemmatizer()
    cleaned_words=[]
    for i in words:
        cleaned_words.append(wordnet_lemmatizer.lemmatize(i))
    #cleaned_words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    Cleanedwords_Upper=[]
    for i in cleaned_words:
        Cleanedwords_Upper.append(i.upper())
    lemmatized_words=[]
    for i in cleaned_words:
        if(i.isalpha()):
            lemmatized_words.append(i)
    LemmatizedUpper=[]
    for i in lemmatized_words:
        LemmatizedUpper.append(i.upper())
    return lemmatized_words

def GetSortedList(ls):
    l=[]
    for i in ls:
        p=i.split('.')
        l.append(p[0])
    l.sort()

    return l

def GetFreqDict(ls):
    freq=[]
    sorted_freq=[]
    for p in ls:
        word_freq = ls.count(p) 
        freq.append(word_freq)
        sorted_freq.append(word_freq)
    sorted_freq.sort()
    return dict(list(zip(ls,freq)))

#calculating Dot product
def Dot_Product(dict1, dict2):  
    S = 0.0
    ExDict1=copy.deepcopy(dict1)
    ExDict2=copy.deepcopy(dict2)
    for k in ExDict1: 
        if k in ExDict2: 
            S =S + (ExDict1[k] * ExDict2[k])               
    return S

def DotProduct_TwoDictionaries(dict1,dict2):
    sum=0
    Exdict1=copy.deepcopy(dict1)
    Exdict2=copy.deepcopy(dict2)

    for i in Exdict1:
        if i in Exdict2:
            sum=sum+(Exdict1[i]*Exdict1[i])

def List_To_Dictionary(ls):
    freq=[]
    list_freq=[]
    for p in ls:
        word_freq = ls.count(p) 
        freq.append(math.log(1+word_freq, 10))
        list_freq.append(1+math.log(word_freq,10))
    list_freq.sort()
    return dict(list(zip(ls,freq)))

#calculating angle between them
def Angle(dict1, dict2):  
    Exdict1=copy.deepcopy(dict1)
    Exdict2=copy.deepcopy(dict2)
    num = Dot_Product(D1, D2) 
    den = math.sqrt(Dot_Product(Exdict1, Exdict1)*Dot_Product(Exdict2, Exdict2))   
    return (num / den) 
    
def Similarity_Score(D1, D2):  
    distance = Angle(D1, D2)   
    return distance

def Similarity(dict1,dict2):
    Exdict1=copy.deepcopy(dict1)
    Exdict2=copy.deepcopy(dict2)
    Dist=Angle(Exdict1,Exdict2)

    return Dist

def Result(modified_tf_idf,result):
    ResultantLength=-1
    if(len(modified_tf_idf)>=10):
        ResultantLength=10
    else:
        ResultantLength=len(modified_tf_idf)
    if(len(modified_tf_idf)>=10):
            length=10
    else:
        length=len(modified_tf_idf)
    for i in range(length):
        result.write("{0} ".format(modified_tf_idf[i]) )
    result.write("\n") 

os.chdir(r'../Dataset/Dataset')
Document_Frequency = defaultdict(int)

DF=dict()
#creating inverse document frequency
for file in glob.iglob('*.html'):
    lemmatized_words=manipulate_file(file)
    for word in list(set(lemmatized_words)):
            Document_Frequency[word] += 1
    for i in list(set(lemmatized_words)):
        if i not in DF:
            DF[i]=1
        else:
            DF[i]=DF[i]+1


FrequencyList=[]
IDF = dict()
for word in Document_Frequency:
    FrequencyList.append(math.log(1000/Document_Frequency[word],10))
    IDF[word] = math.log(1000 / float(Document_Frequency[word]),10)
#print(len(IDF))

term_Freq=dict()
tf_idf = dict()
Inverted_Positional_Index = dict()
tf_Dict = dict()
Local_champions = dict()
total_words= []
tfDict_new = dict()
Global_champions = dict()
tf_idf_doc=dict()
index=-1

for file in glob.iglob('*.html'):
    index+=1
    lemmatized_words=manipulate_file(file)

    temp=len(lemmatized_words)
    i=0
    while(i<temp):
        temp2=lemmatized_words[i]
        if temp2 not in total_words:
            total_words.append(temp2)
        i=i+1

     
    #calculating term frequency       
    temp = List_To_Dictionary(lemmatized_words)    
    for i in temp.keys():
        term_Freq[(i,file)]=temp[i]
    temp2=copy.deepcopy(temp)
    Frequencies=[]
    for j in temp2.keys():
        Frequencies.append(temp2[j])
    
    j=0
    l=list(set(lemmatized_words))
    temp3=len(set(lemmatized_words))
    while(j<temp3):
        tf_idf[(l[j],file)]=term_Freq[(l[j],file)]*IDF[l[j]]
        j=j+1 
    
    l=list(set(lemmatized_words))
    temp4=len(l)
    i=0
    while(i<temp4):
        Inverted_Positional_Index[(l[i],IDF[l[i]])] = (file,term_Freq[(l[i],file)])
        i=i+1
    
    l=list(set(lemmatized_words))
    temp5=len(l)
    i=0
    while(i<temp5):
        t=l[i]
        if t not in tf_Dict.keys():
            tf_Dict[t]=[]
        tf_Dict[t].append((file,term_Freq[(t,file)]))
        i=i+1


    l=list(set(lemmatized_words))
    temp6=len(l)
    j=0
    while(j<temp6):
        t1=l[j]
        if t1 not in tfDict_new.keys():
            tfDict_new[t1]=[]
        tfDict_new[t1].append((file,tf_idf[(t1,file)]+static_quality_score[index]))
        j=j+1

#print(len(tf_idf))
#Generating local champions
for word in list(set(total_words)):
    tf_Dict[word].sort(key = lambda x:x[1],reverse=True)


for word in list(set(total_words)):
        NList=[]
        if word not in Local_champions.keys():
            Local_champions[word] = list()
        newList = []
        if(len(tf_Dict[word])>=50):
            length=50
        else:
            length=len(tf_Dict[word])
        newList=tf_Dict[word][:length]
        Nlist=copy.deepcopy(newList)
        t7=""
        j=0
        for i in Nlist:
            t7=t7+str(j)
            j=j+1
        for x in newList:
            Local_champions[word].append(x[0])

#Generating global champions
for word in set(total_words):
    tfDict_new[word].sort(key = lambda x:x[1],reverse=True)


for word in set(total_words):
        if word not in Global_champions.keys():
            Global_champions[word] = list()
        newList = []
        if(len(tfDict_new[word])>=50):
            length=50
        else:
            length=len(tfDict_new[word])
        newList=tfDict_new[word][:length]
        for x in newList:
            Global_champions[word].append(x[0])

os.chdir('..')
os.chdir(r'../code')
result = open("RESULTS2_20CS60R56.txt","w+")
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

os.chdir(r'../Dataset/Dataset')
for l in lines:
    l = l.lower() 
    tokens = word_tokenize(l)    
    words=[]
    for i in tokens:
    	if(i.isalpha()):
    		words.append(i)
    words = [w for w in words if not w in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    Q=[]
    for i in words:
    	Q.append(wordnet_lemmatizer.lemmatize(i))
    #print(len(Q))

    tf_idf_dictionary = {}
    Local_dictionary = {}
    Global_dictionary = {}
    Cluster_dictionary = {}

    #tf_idf scoring
    result.write(l)
    result.write('\n')
    ind=-1
    for file in glob.iglob('*.html'):
        try:
            tf_idf_dictionary[file]=score(file,Q)
        except:
            pass   
    modified_tf_idf = sorted(tf_idf_dictionary.items(), key=operator.itemgetter(1),reverse=True)
    #print(modified_tf_idf)

    #Cluster pruning scheme
    for file in leaders:
        try:
            Cluster_dictionary[file]=score(file,Q)
        except:
            pass
    try:
        leader = max(Cluster_dictionary, key= lambda x: Cluster_dictionary[x]) 
        leader_cleaned_words=manipulate_file(leader)
        ListLower=[]
        for i in leader_cleaned_words:
            ListLower.append(i)
        for i in leader_cleaned_words:
            i=i.lower()
        tf_leader = GetFreqDict(leader_cleaned_words)
        ListUpper=[]
        for i in leader_cleaned_words:
            ListUpper=[]
        for i in leader_cleaned_words:
            i=i.upper()


        follower_dict=dict()
        for file in leaders:
            if leader!=file:
                follower_cleaned_words=manipulate_file(file)
                tf_follower = GetFreqDict(follower_cleaned_words)
                follower_dict[file]=Similarity_Score(tf_leader,tf_follower)
        follower_dict=sorted(follower_dict.items(), key=operator.itemgetter(1),reverse=True)

        ''' selecting top 10 followers '''
        follower=[]
        follower.append(leader)
        for i in range(9):
            follower.append(follower_dict[i][0])

        UpdatedFollowers=list()
        tt=9
        k=0
        while(k<t):
            UpdatedFollowers.append(follower_dict[i][0])
            k=k+1
        UpdatedFollwers.append(leader)

        UpdatedFollwers.sort()

        Cluster_dictionary_new=dict()
        sorted_Cluster_new=[]
        for file in follower:
            try:
                Cluster_dictionary_new[file]=score(file,Q)
            except:
                pass
        sorted_Cluster_new = sorted(Global_dictionary.items(), key=operator.itemgetter(1),reverse=True)
    except:
        pass

    #Global Champion scoring
    d_global=[]
    for term in Q:
        try:
            for i in Global_champions[term]:
                d_global.append(i)
        except:
            pass
    for file in d_global:
        try:
            Global_dictionary[file]=score(file,Q)
        except:
            pass
    sorted_Global = sorted(Global_dictionary.items(), key=operator.itemgetter(1),reverse=True)

    #local Champion Scoring
    local=[]
    for term in Q:
        try:
            for i in Local_champions[term]:
                local.append(i)
        except:
            pass    
    for file in local:
        try:
            Local_dictionary[file]=score(file,Q)
        except:
            pass
    sorted_Local = sorted(Local_dictionary.items(), key=operator.itemgetter(1),reverse=True)




    try:
        Result(sorted_Cluster_new,result)
    except:
        pass
    Result(modified_tf_idf,result)
    Result(sorted_Local,result)
    Result(sorted_Global,result)
  
    result.write("\n")
