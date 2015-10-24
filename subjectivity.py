from tsvfiles import tsvreader
from features import features_subjectivity
from tokenizers import twokenize
from postaggers import arktagger 
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams
from lexicons import negations,Slang,SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
from clusters import Clusters
from utilities import *
import learningCurves
import time

#def classify(train,test):

start_time = time.time()

#load training set
dataset_train = "datasets/training-set-sample.tsv"
#dataset_train = "datasets/train15.tsv"
#dataset_train = train
labels_train, messages_train = tsvreader.opentsv(dataset_train)

#load testing set
#dataset_test = "datasets/dev15.tsv"
dataset_test = "datasets/testing-set-sample.tsv"
#dataset_test = test
labels_test, messages_test = tsvreader.opentsv(dataset_test)

#load Slang Dictionary
slangDictionary = Slang.Slang()

#merge positive-negative categories into one category(subjective), as we
#want to check the subjectivity of message
# 0 - objective(neutral) messages
# 1 - subjective(positive or negatve) messages
labels_train = [0 if x=="neutral" else 1 for x in labels_train]
labels_test = [0 if x=="neutral" else 1 for x in labels_test]

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)

#compute pos tag bigrams for all messages
pos_bigrams_train = getBigrams(pos_tags_train)
pos_bigrams_test = getBigrams(pos_tags_test)

#get the unique pos bigrams from training set
unique_bigrams = getBigramsSet(pos_bigrams_train)
#calculate pos bigrams score for all categories
#both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
pos_bigrams_scores_objective = posBigramsScore(unique_bigrams,0,pos_bigrams_train,labels_train)
pos_bigrams_scores_subjective = posBigramsScore(unique_bigrams,1,pos_bigrams_train,labels_train)

#Load Lexicons

#Minqing Hu Lexicon
minqinghu = MinqingHuLexicon.MinqingHuLexicon()
#Afinn Lexicon
afinn = Afinn()
#NRC Lexicons
nrc2 = NRCLexicon.NRCLexicon(1)
nrc5 = NRCLexicon.NRCLexicon(4)
nrc6 = NRCLexicon.NRCLexicon(5)
#SentiWordNet Lexicon
swn = SentiWordNetLexicon.SentiWordNetLexicon(False)
#SentiWordNet Lexicon - AverageScores
swn_avg= SentiWordNetLexicon.SentiWordNetLexicon(True)

#do not include MPQA Lexicons
lexicons = [minqinghu,afinn,nrc2,nrc5,nrc6,swn,swn_avg]

#MPQA Lexicons (8 Lexicons)
S_pos = MPQALexicon.MPQALexicon(0)
S_neg = MPQALexicon.MPQALexicon(1)
S_pos_neg = MPQALexicon.MPQALexicon(2)
S_neu = MPQALexicon.MPQALexicon(3)
W_pos = MPQALexicon.MPQALexicon(4)
W_neg = MPQALexicon.MPQALexicon(5)
W_pos_neg = MPQALexicon.MPQALexicon(6)
W_neu = MPQALexicon.MPQALexicon(7)

mpqa_lexicons = [S_pos,S_neg,S_pos_neg,S_neu,W_pos,W_neg,W_pos_neg,W_neu]

#assign a precision and F1 score to each word of to all mpqa lexicons
mpqaScores = getScores(mpqa_lexicons,messages_train,labels_train)

#get negations list
negationList = negations.loadNegations();

#load word clusters
clusters = Clusters.Clusters()

#get features from train messages
features_train = features_subjectivity.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_train,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)
#get features from test messages 
features_test = features_subjectivity.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

#train classifier and return trained model
model = LogisticRegression.train(features_train,labels_train)
#model = SVM.train(features_train,labels_train)

#predict labels
prediction = LogisticRegression.predict(features_test,model)
#prediction = SVM.predict(features_test,model)

#calculate accuracy
print "Average F1 : " +str(measures.avgF1(labels_test,prediction))
print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
print "F1 Objective : " +str(measures.F1(labels_test,prediction,0))
print "F1 Subjective : " +str(measures.F1(labels_test,prediction,1))
print "Precision Objective: " +str(measures.precision(labels_test,prediction,0))
print "Precision Subjective: " +str(measures.precision(labels_test,prediction,1))
print "Recall Objective : " +str(measures.recall(labels_test,prediction,0))
print "Recall Subjective : " +str(measures.recall(labels_test,prediction,1))




#plot learning curve
#learningCurves.plot_learning_curve(len(messages_train),features_train,labels_train,features_test,labels_test)

print("--- %s seconds ---" % (time.time() - start_time))
