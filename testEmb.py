from lexicons import GloveDictionary
from tokenizers import twokenize
import numpy as np
from tsvfiles import tsvreader
from utilities import *
from classifiers import LogisticRegression
from evaluation import measures

def findCentroid(embeddings):
    centroid= []

    for j in range(0,len(embeddings[0])):

        f = embeddings[:,j]
        s=0

        for i in range(0,len(f)):
           s+=float(f[i])

        centroid.append(s/float(len(f)))

    return centroid

#load the data
#dataset_train = "datasets/train15.tsv"
#dataset_test = "datasets/dev15.tsv"

dataset_train = "datasets/training-set-sample.tsv"
dataset_test = "datasets/testing-set-sample.tsv"

labels_train,messages_train=tsvreader.opentsv(dataset_train)
labels_test, messages_test =tsvreader.opentsv(dataset_test)

labels_train = [0 if x=="neutral" else 1 for x in labels_train]
labels_test = [0 if x=="neutral" else 1 for x in labels_test]

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#initialize glove lexicon
glove = GloveDictionary.Glove()

print("glove initialized ... " )

features_train = []

for message in tokens_train :

    embeddings=[]
    
    for token in message : 
        embeddings.append(glove.findWordEmbeddings(token))

    features_train.append(findCentroid(np.asarray(embeddings)))        

print("Train Embeddings created ...." )

features_test=[]

for message in tokens_test :

    embeddings=[]
    
    for token in message : 
        embeddings.append(glove.findWordEmbeddings(token))

    features_test.append(findCentroid(np.asarray(embeddings)))


print("Test Embeddings created ...." )

model = LogisticRegression.train(features_train,labels_train)

prediction = LogisticRegression.predict(features_test,model)

print "Average F1 : " +str(measures.avgF1(labels_test,prediction,0,1))
print "Accuracy : " +str(measures.accuracy(labels_test,prediction))

print(prediction)

