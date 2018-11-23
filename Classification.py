from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np 
from itertools import cycle
from pandas import DataFrame
from sklearn.cross_validation import StratifiedKFold
import gensim


vectorizer = CountVectorizer(stop_words='english')
df = pd.read_csv("train_set.csv",sep="\t")
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y = le.transform(df["Category"])
X = vectorizer.fit_transform(df['Content'])



vocab = vectorizer.get_feature_names()
corpus_vect_gensim = gensim.matutils.Sparse2Corpus(X, documents_columns=False)




# Topics = 10


lda = gensim.models.ldamodel.LdaModel(corpus_vect_gensim, id2word=dict([(i, s) for i, s in enumerate(vocab)]), num_topics=10)


doc_topic = lda.get_document_topics(corpus_vect_gensim)

mod_X = gensim.matutils.corpus2csc(doc_topic)

New_X = np.transpose(mod_X)



print("MultinomialNB")
clf = MultinomialNB()
scores_multi = cross_val_score(clf, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_multi)


print("BernoulliNB")
clf2 = BernoulliNB()
scores_bernouli = cross_val_score(clf2, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_bernouli)


print("SVC")
clf3 = svm.SVC(C=1.0)
scores_svc = cross_val_score(clf3, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_svc)


print("Random Forests")
clf4 = RandomForestClassifier(n_estimators=100)
scores_rf = cross_val_score(clf4, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_rf)


print("KNearest Neighbor")
clf5 = KNeighborsClassifier(n_neighbors=5)
scores_knn = cross_val_score(clf5, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_knn)


# Our method

k_range = range(1,20)
k_scores = []
for k in k_range :
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, New_X, Y , cv=10 , scoring='accuracy')
    k_scores.append(scores.mean())



maximun = max(k_scores)
maxIndex = k_scores.index(maximun)
maxIndex += 1
maxIndex

print("My Method")
clf6 = KNeighborsClassifier(n_neighbors=maxIndex)
scores_my_knn = cross_val_score(clf6, New_X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_my_knn)




# Topics = 100



lda2 = gensim.models.ldamodel.LdaModel(corpus_vect_gensim, id2word=dict([(i, s) for i, s in enumerate(vocab)]), num_topics=100)


doc_topic2 = lda2.get_document_topics(corpus_vect_gensim)

mod_X2 = gensim.matutils.corpus2csc(doc_topic2)

New_X2 = np.transpose(mod_X2)



print("MultinomialNB")
clf = MultinomialNB()
scores_multi2 = cross_val_score(clf, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_multi2)


print("BernoulliNB")
clf2 = BernoulliNB()
scores_bernouli2 = cross_val_score(clf2, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_bernouli2)


print("SVC")
clf3 = svm.SVC(C=1.0)
scores_svc2 = cross_val_score(clf3, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_svc2)


print("Random Forests")
clf4 = RandomForestClassifier(n_estimators=100)
scores_rf2 = cross_val_score(clf4, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_rf2)


print("KNearest Neighbor")
clf5 = KNeighborsClassifier(n_neighbors=5)
scores_knn2 = cross_val_score(clf5, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_knn2)


# Our method

k_range = range(1,20)
k_scores = []
for k in k_range :
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, New_X2, Y , cv=10 , scoring='accuracy')
    k_scores.append(scores.mean())



maximun2 = max(k_scores)
maxIndex2 = k_scores.index(maximun2)
maxIndex2 += 1
maxIndex2

print("My Method")
clf6 = KNeighborsClassifier(n_neighbors=maxIndex2)
scores_my_knn2 = cross_val_score(clf6, New_X2, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_my_knn2)





# Topics = 1000


lda3 = gensim.models.ldamodel.LdaModel(corpus_vect_gensim, id2word=dict([(i, s) for i, s in enumerate(vocab)]), num_topics=1000)


doc_topic3 = lda3.get_document_topics(corpus_vect_gensim)

mod_X3 = gensim.matutils.corpus2csc(doc_topic3)

New_X3 = np.transpose(mod_X3)



print("MultinomialNB")
clf = MultinomialNB()
scores_multi3 = cross_val_score(clf, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_multi3)


print("BernoulliNB")
clf2 = BernoulliNB()
scores_bernouli3 = cross_val_score(clf2, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_bernouli3)


print("SVC")
clf3 = svm.SVC(C=1.0)
scores_svc3 = cross_val_score(clf3, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_svc3)


print("Random Forests")
clf4 = RandomForestClassifier(n_estimators=100)
scores_rf3 = cross_val_score(clf4, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_rf3)


print("KNearest Neighbor")
clf5 = KNeighborsClassifier(n_neighbors=5)
scores_knn3 = cross_val_score(clf5, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_knn3)


# Our method

k_range = range(1,20)
k_scores = []
for k in k_range :
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, New_X3, Y , cv=10 , scoring='accuracy')
    k_scores.append(scores.mean())



maximun3 = max(k_scores)
maxIndex3 = k_scores.index(maximun3)
maxIndex3 += 1
maxIndex3

print("My Method")
clf6 = KNeighborsClassifier(n_neighbors=maxIndex3)
scores_my_knn3 = cross_val_score(clf6, New_X3, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_my_knn3)




multi_pd = [scores_multi , scores_multi2 , scores_multi3]
bernouli_pd = [scores_bernouli , scores_bernouli2 , scores_bernouli3]
svc_pd = [scores_svc , scores_svc2 , scores_svc3]
rf_pd = [scores_rf , scores_rf2 , scores_rf3]
knn_pd = [scores_knn , scores_knn2 , scores_knn3]
my_knn_pd = [scores_my_knn , scores_my_knn2 , scores_my_knn3]


new_df = DataFrame({'MultinomialNB':multi_pd , 'BernouliNB':bernouli_pd ,
                    'SVC(SVM)':svc_pd , 'Random Forests':rf_pd ,
                    'K-nearest neighbors':knn_pd , 'My Method':my_knn_pd} , index=['Accuracy K = 10' , 'Accuracy K = 100' , 'Accuracy K = 1000'])

new_df.to_csv("EvaluationMetric_10fold_ida_only.csv", sep='\t')






# Best method - testSet_categories


vectorizer2 = CountVectorizer(stop_words='english')
df2 = pd.read_csv("test_set.csv",sep="\t")
pred_X = vectorizer2.fit_transform(df2['Content'])


vocab2 = vectorizer2.get_feature_names()
corpus_vect_gensim2 = gensim.matutils.Sparse2Corpus(pred_X, documents_columns=False)

lda4 = gensim.models.ldamodel.LdaModel(corpus_vect_gensim2, id2word=dict([(i, s) for i, s in enumerate(vocab2)]), num_topics=100)

doc_topic4 = lda4.get_document_topics(corpus_vect_gensim2)

New_pred_X = gensim.matutils.corpus2csc(doc_topic4)

New_pred_X2 = np.transpose(New_pred_X)


clf6 = KNeighborsClassifier(n_neighbors=maxIndex2)
clf6.fit(New_X2,Y)
my_method = clf6.predict(New_pred_X2)


my_id = df2['Id']

temp_id = []
temp_cat = []

for i in range(len(my_id)) :
    cat = ""
    if(my_method[i] == 0) :
        cat = "Business"
    if(my_method[i] == 1) :
        cat = "Film"
    if(my_method[i] == 2) :
        cat = "Football"
    if(my_method[i] == 3) :
        cat = "Politics"
    if(my_method[i] == 4) :
        cat = "Technology"
    temp_id.append(my_id[i])
    temp_cat.append(cat)
                           
df2 = DataFrame({'Id':temp_id , 'Predicted category':temp_cat})

df2.to_csv("testSet_categories.csv", sep='\t')


