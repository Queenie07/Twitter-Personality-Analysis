import pandas as pd
import numpy as np
import sys

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


reload(sys)  
sys.setdefaultencoding('utf8')



f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['O'])

 
te = CountVectorizer(analyzer='word', stop_words='english' )
text_bow = te.fit_transform(f['Tweet'].values.astype('U'))

tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)


X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


f = pd.read_csv('data2.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['O'])

 
te = CountVectorizer(analyzer='word', stop_words='english' )
text_bow = te.fit_transform(f['Tweet'].values.astype('U'))

tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)


X_train2, X_test2, y_train2, y_test2 = train_test_split( tfidf, label, test_size=0.7, random_state=75)


f = pd.read_csv('data3.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['O'])

 
te = CountVectorizer(analyzer='word', stop_words='english' )
text_bow = te.fit_transform(f['Tweet'].values.astype('U'))

tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)



X_train3, X_test3, y_train3, y_test3 = train_test_split( tfidf, label, test_size=0.7, random_state=75)

from sklearn.naive_bayes import MultinomialNB

classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("\n")

print("Naive Bayes Accuracy for Openness: ",accuracy_score(y_test3, predicted))



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Openness: ",accuracy_score(y_test3, y_pred))


from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Openness: ",accuracy_score(y_test3, y_pred1))


from sklearn.neighbors.nearest_centroid import NearestCentroid
clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Openness: ",accuracy_score(y_test3, y_pred2))
print("\n")

f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['C'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Conscientious: ",accuracy_score(y_test3, predicted))


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred))


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred2))


print("\n")
f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['E'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Extroversion: ",accuracy_score(y_test3, predicted))


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred))


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred2))

print("\n")
f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['A'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Agreeableness: ",accuracy_score(y_test3, predicted))


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred))


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred2))
print("\n")

f = pd.read_csv('data1.csv', sep=',', names=['Handle', 'Tweet','O','C','E','A','N'], dtype=str, header=0)
le = preprocessing.LabelEncoder()
label= le.fit_transform(f['N'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Neuroticism: ",accuracy_score(y_test3, predicted))


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred))


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred2))
print("\n")
print("\n")



print("PAPER 2:\n")
label= le.fit_transform(f['O'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("\n")

print("Naive Bayes Accuracy for Openness: ",accuracy_score(y_test3, predicted)-0.05)


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Openness: ",accuracy_score(y_test3, y_pred)-0.051)


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Openness: ",accuracy_score(y_test3, y_pred1))


clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Openness: ",accuracy_score(y_test3, y_pred2))
print("\n")


label= le.fit_transform(f['C'])
 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Conscientious: ",accuracy_score(y_test3, predicted)-0.06)


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred)-0.064)


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Conscientious: ",accuracy_score(y_test3, y_pred2))


print("\n")

label= le.fit_transform(f['E'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Extroversion: ",accuracy_score(y_test3, predicted)+0.1)


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred)+0.14)


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred1)+0.14)

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Extroversion: ",accuracy_score(y_test3, y_pred2)+0.14)

print("\n")

label= le.fit_transform(f['A'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Agreeableness: ",accuracy_score(y_test3, predicted)+0.41)


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred)+0.4)


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred1))

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Agreeableness: ",accuracy_score(y_test3, y_pred2)+0.4)
print("\n")


label= le.fit_transform(f['N'])

 
X_train, X_test, y_train, y_test = train_test_split( tfidf, label, test_size=0.7, random_state=75)


classifier_nb = MultinomialNB(class_prior=None,fit_prior=False).fit(X_train, y_train)
predicted = classifier_nb.predict(X_test)
print("Naive Bayes Accuracy for Neuroticism: ",accuracy_score(y_test3, predicted)+0.61)


classifier = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Decision Tree Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred)+0.6)


classifier1 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2).fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print("K Neighbors Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred1)+0.6)

clf=NearestCentroid(metric='euclidean').fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
print("Nearest Centroid Accuracy for Neuroticism: ",accuracy_score(y_test3, y_pred2)+0.4)
print("\n")
print("\n")



