from bs4 import BeautifulSoup
import glob
import os
import re
import contextlib
import pickle

path = os.getcwd()
os.chdir(path)

os.mkdir('ECTText')
os.mkdir('ECTNestedDict')
count=[]
months=['January','Febraury','March','April','May','June','July','August','September','October','November','December']
os.chdir(path+'/ECT')
for file in glob.iglob('*.html'):
	FileNames = [file for file in glob.glob("*.html")]
	FileNames.sort()
	FileIndex=FileNames.index(file)
	dict1={"Date":"","Participants":[],"Presentation":{},"Questionnaire":{}}
	file_name=file;
	#print(file_name)
	li=file_name.split('.')
	#count.append(li[0])
	F=open(file,encoding='utf8')
	soup = BeautifulSoup(F,'html5lib')
	string=soup.p.get_text()
	#print(string)
	date=""
	result=0
	#print(months)
	for i in months:
		
		if(string.find(i)>0):
			result=string.find(i)
			#print(result)
			date=date+string[result:]
	#print(date)
	dict1["Date"]=date
	#print(dict1)

	names_of_participants=[]
	list1=soup.find_all('p',class_='p p1')
	for i in list1:
		text=i.get_text()
		if(len(text)<=80 and (text.find('-'))>2 and text!="Question-and-Answer Session"):
			names_of_participants.append(text)
	dict1["Participants"]=names_of_participants
	#print(dict1)


	Participants=[]
	temp=""
	for i in names_of_participants:
		l=i.split('-')
		if(len(l)==2):
			temp=l[0].strip()
		else:
			temp=l[0]+'-'+l[1]
			temp=temp.strip()
		Participants.append(temp)
	#print(Participants)


	speech_of_participant=""
	name_of_the_participant=""
	list1=soup.find_all('p')
	speech={}

	#for i in list1:
	#	print(i.get_text()) 

	for i in range(len(list1)):
		temp=list1[i].get_text()
		
		if(temp=="Question-and-Answer Session"):
			break
		if temp in Participants or temp=="Operator" :
			#print(temp)
			speech_of_participant=""
			name_of_the_participant=temp
			#print(temp)
			for j in range(i+1,len(list1)):
				#print(list1[j].get_text())
				if(list1[j].get_text() in Participants or list1[j].get_text()=="Operator" or list1[j].get_text()=="Question-and-Answer Session"):
					#print(name_of_the_participant)
					#print(speech_of_participant)
					if(len(name_of_the_participant)>0 and len(speech_of_participant)>0):
						speech[name_of_the_participant]=speech_of_participant
					#i=j
					break
				else:
					speech_of_participant=speech_of_participant+list1[j].get_text()

		#speech[name_of_the_participant]=speech_of_participant

	#print(speech)
	dict1["Presentation"]=speech
	#print(dict1)
	#print("\n")
	#print("\n")
	#print("\n")

	count=0
	questions={}
	#string3=soup.find('p',id="question-answer-session")
	#print(file_name)
	t=0
	for j in range(len(list1)):
		if(list1[j].get_text()=="Question-and-Answer Session"):
			t=j
			break
	for i in range(t+1,len(list1)):
		temp=list1[i].get_text()
		mmm=temp.split('-')
		nnn=mmm[-1]
		ppp=nnn.strip()
		if temp in Participants or temp=="Operator" or ppp in Participants :
			
			#print(temp)
			speaker={}
			question_of_participant=""
			name_of_the_participant=temp
			#print(temp)
			for j in range(i+1,len(list1)):
				#print(list1[j].get_text())
				if(list1[j].get_text() in Participants or list1[j].get_text()=="Operator" ):
					#print(name_of_the_participant)
					#print(speech_of_participant)
					if(len(name_of_the_participant)>0 and len(speech_of_participant)>0):
						count=count+1
						speaker[name_of_the_participant]=question_of_participant
						questions[count]=speaker
					#i=j
						break
				else:
					question_of_participant=question_of_participant+list1[j].get_text()
	#print(file_name)
	#print(questions)
	dict1["Questionnaire"]=questions

	#for i in list2:
	#	print(i)
	#print(dict1)'''

	pickle_out = open(path+"\\ECTNestedDict\\"+str(FileIndex)+'.pickle',"wb")
	pickle.dump(dict1, pickle_out)
	pickle_out.close()

os.chdir(path+"/ECT")
for file in glob.iglob('*.html'):
	FileNames = [file for file in glob.glob("*.html")]
	FileNames.sort()
	FileIndex=FileNames.index(file)
	F=open(file,encoding='utf8')
	file_name=file;
	#print(file_name)
	li=file_name.split('.')
	soup = BeautifulSoup(F,'html5lib')
	string=soup.get_text()
	string=string.lower()
	with open(path+"/ECTText/"+str(FileIndex)+'.txt','w',encoding='utf8',) as file:
		file.write(string)
