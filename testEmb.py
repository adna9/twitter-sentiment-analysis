from embeddings import GloveDictionary
from tokenizers import twokenize
import numpy as np
from tsvfiles import tsvreader
from utilities import *
from classifiers import LogisticRegression
from evaluation import measures
from utilities import *

#load the data
dataset_train = "datasets/train15.tsv"
dataset_test = "datasets/dev15.tsv"

#dataset_train = "datasets/training-set-sample.tsv"
#dataset_test = "datasets/testing-set-sample.tsv"

labels_train,messages_train=tsvreader.opentsv(dataset_train)
labels_test, messages_test =tsvreader.opentsv(dataset_test)

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#initialize glove lexicon
glove = GloveDictionary.Glove()

##labels_train_sub = [0 if x=="neutral" else 1 for x in labels_train]
##labels_test_sub = [0 if x=="neutral" else 1 for x in labels_test]
##
##print("glove initialized ... " )
##
##features_train = []
##for message in tokens_train :
##    features_train.append(glove.findCentroid(message))
##                          
##features_train = np.array(features_train)
##
##print("Train Embeddings created ...." )
##
##features_test=[]
##
##for message in tokens_test :
##    features_test.append(glove.findCentroid(message))
##
##features_test = np.array(features_test)
##
##print("Test Embeddings created ...." )
##
##model = LogisticRegression.train(features_train,labels_train)
##
##prediction = LogisticRegression.predict(features_test,model)
##
##print "Average F1 : " +str(measures.avgF1(labels_test,prediction,0,1))
##print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
##
###print(prediction)

