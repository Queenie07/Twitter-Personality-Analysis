import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



#import csv,txt
#output=open('xyz.csv','w')
#with open('abc.txt',"rt", encoding='ascii') as f:
#for row in f:
#    output.write(row)



f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['O'])
print (le.classes_)
 
te = CountVectorizer(analyzer='word', stop_words='english' )
text_bow = te.fit_transform(f['Tweet'].values.astype('U'))

tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)


X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)



from sklearn.naive_bayes import MultinomialNB

classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)

cm1=confusion_matrix(y_test, predicted)
print(cm1)
total1=sum(sum(cm1))


sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy  Naive: ', accuracy1)



print('Recall Naive: ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity  Naive: ', specificity1)
precision = cm1[0,0]/(cm1[0,0]+cm1[1,0])
print('Precision  Naive: ', precision)

Fmeasure=(2*precision*sensitivity1)/(precision+sensitivity1)
print(Fmeasure)

score = classifier_nb.score(X_test,y_test)
print(score)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %f)' % roc_auc)
plt.plot([0, 1], [0, 1],linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend()
plt.show()

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm1=confusion_matrix(y_test, y_pred)

from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
cm1=confusion_matrix(y_test, y_pred1)

#ROCCHIO

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
cm1=confusion_matrix(y_test, y_pred2)
print(cm1)


