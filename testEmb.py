from embeddings import GloveDictionary
from tokenizers import twokenize
import numpy as np
from tsvfiles import tsvreader
from utilities import *
from classifiers import LogisticRegression,SVM
from evaluation import measures
from utilities import *
from postaggers import arktagger
import enchant
from lexicons import Slang
from preProcess import *
import learningCurves

#load the data
dataset_train = "datasets/train15.tsv"
dataset_test = "datasets/dev15.tsv"

#dataset_train = "datasets/training-set-sample.tsv"
#dataset_test = "datasets/testing-set-sample.tsv"

labels_train,messages_train=tsvreader.opentsv(dataset_train)
labels_test, messages_test =tsvreader.opentsv(dataset_test)

##labels_train,messages_train=tsvreader.opentsvPolarity(dataset_train)
##labels_test, messages_test =tsvreader.opentsvPolarity(dataset_test)

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#initialize glove lexicon
glove = GloveDictionary.Glove()

#dictionary = enchant.Dict("en_US")

#slangDictionary = Slang.Slang()

labels_train = [0 if x=="neutral" else 1 for x in labels_train]
labels_test = [0 if x=="neutral" else 1 for x in labels_test]

##labels_train = [0 if x=="negative" else 1 for x in labels_train]
##labels_test = [0 if x=="negative" else 1 for x in labels_test]


pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)


##messages_train = preprocessMessages(messages_train,tokens_train,pos_tags_train,slangDictionary,dictionary)
##messages_test = preprocessMessages(messages_test,tokens_test,pos_tags_test,slangDictionary,dictionary)
##
##tokens_train=tokenize(messages_train)
##tokens_test=tokenize(messages_test)
##
###compute pos tags for all preprocessed messages
##pos_tags_train = arktagger.pos_tag_list(messages_train)
##pos_tags_test = arktagger.pos_tag_list(messages_test)


print("glove initialized ... " )

features_train = []
#for message in tokens_train :
for i in range(0,len(messages_train)):
    features_train.append(glove.findCentroid(messages_train[i],tokens_train[i],pos_tags_train[i]))
                          
features_train = np.array(features_train)

print("Train Embeddings created ...." )

features_test=[]

#for message in tokens_test :
for i in range(0,len(messages_test)):
    features_test.append(glove.findCentroid(messages_test[i],tokens_test[i],pos_tags_test[i]))

features_test = np.array(features_test)

print("Test Embeddings created ...." )

##model = LogisticRegression.train(features_train,labels_train)
##prediction = LogisticRegression.predict(features_test,model)

model = SVM.train(features_train,labels_train,k="linear")
prediction = SVM.predict(features_test,model)


print "Average F1 : " +str(measures.avgF1(labels_test,prediction,0,1))
print "Accuracy : " +str(measures.accuracy(labels_test,prediction))


learningCurves.plot_learning_curve(features_train,labels_train,features_test,labels_test,C=1)

