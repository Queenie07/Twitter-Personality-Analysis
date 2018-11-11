###############
# CSV to ARFF #
###############
import csv
from time import sleep

import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer




reload(sys)  
sys.setdefaultencoding('utf8')


content=[]
f=open('data1O.arff','a')
f1=open('data1C.arff','a')
with open('data1.csv', 'rb') as csvfile:
	lines = csv.reader(csvfile, delimiter = ',')
        for row in lines:
                   if(row[0]!='Handle'):
			text=row[1]
  			tokens = word_tokenize(text)
  			doc = []
  			stop_words = set(stopwords.words('english'))
  			for w in tokens:
   			 try:
   			  if w not in stop_words:
			        doc.append(w)

   			 except UnicodeDecodeError:
				print("")

  			ps = PorterStemmer()
  			l=[]
  			for w in doc:
  			 try:
  			  l.append(ps.stem(w))
  			 except UnicodeDecodeError:
  			  print("")
	
			f.write(row[0])
		    	f.write(',')
		    	f.write(row[2])
		    	f.write('\n')
	
			f1.write(row[0])
		    	f1.write(',')
		    	f1.write(row[3])
		    	f1.write('\n')

                    	content.append(l)
                    	content.append(row[2])
		    	content.append('\n')
csvfile.close()
print(content)





content=[]
f=open('data2O.arff','a')
f1=open('data2C.arff','a')
with open('data2.csv', 'rb') as csvfile:
	lines = csv.reader(csvfile, delimiter = ',')
        for row in lines:
                   if(row[0]!='Handle'):
			text=row[1]
  			tokens = word_tokenize(text)
  			doc = []
  			stop_words = set(stopwords.words('english'))
  			for w in tokens:
   			 try:
   			  if w not in stop_words:
			        doc.append(w)

   			 except UnicodeDecodeError:
				print("")

  			ps = PorterStemmer()
  			l=[]
  			for w in doc:
  			 try:
  			  l.append(ps.stem(w))
  			 except UnicodeDecodeError:
  			  print("")
	
			f.write(row[0])
		    	f.write(',')
		    	f.write(row[2])
		    	f.write('\n')
	
			f1.write(row[0])
		    	f1.write(',')
		    	f1.write(row[3])
		    	f1.write('\n')

                    	content.append(l)
                    	content.append(row[2])
		    	content.append('\n')
csvfile.close()
print(content)


content=[]
f=open('data3O.arff','a')
f1=open('data3C.arff','a')
with open('data3.csv', 'rb') as csvfile:
	lines = csv.reader(csvfile, delimiter = ',')
        for row in lines:
                   if(row[0]!='Handle'):
			text=row[1]
  			tokens = word_tokenize(text)
  			doc = []
  			stop_words = set(stopwords.words('english'))
  			for w in tokens:
   			 try:
   			  if w not in stop_words:
			        doc.append(w)

   			 except UnicodeDecodeError:
				print("")

  			ps = PorterStemmer()
  			l=[]
  			for w in doc:
  			 try:
  			  l.append(ps.stem(w))
  			 except UnicodeDecodeError:
  			  print("")
	
			f.write(row[0])
		    	f.write(',')
		    	f.write(row[2])
		    	f.write('\n')
	
			f1.write(row[0])
		    	f1.write(',')
		    	f1.write(row[3])
		    	f1.write('\n')

                    	content.append(l)
                    	content.append(row[2])
		    	content.append('\n')
csvfile.close()
print(content)



