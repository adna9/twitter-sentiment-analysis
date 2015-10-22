from tsvfiles import tsvreader
from features import features_polarity as features
from postaggers import arktagger 
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams
from lexicons import negations,Slang,SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
from clusters import Clusters
import learningCurves
import time
from utilities import *
from feature_selection import selection


#def main():

start_time = time.time()

#load training set
#dataset_train = "datasets/tweets#2013.tsv"
dataset_train = "datasets/train15.tsv"
labels_train, messages_train = tsvreader.opentsvPolarity(dataset_train)

#load testing set
dataset_test = "datasets/dev15.tsv"
#dataset_test = "datasets/devtweets2013.tsv"
labels_test, messages_test = tsvreader.opentsvPolarity(dataset_test)



#load Slang Dictionary
slangDictionary = Slang.Slang()

# 0 - negative messages
# 1 - positives messages
labels_train = [0 if x=="negative" else 1 for x in labels_train]
labels_test = [0 if x=="negative" else 1 for x in labels_test]

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
pos_bigrams_scores_negative = posBigramsScore(unique_bigrams,0,pos_bigrams_train,labels_train)
pos_bigrams_scores_positive = posBigramsScore(unique_bigrams,1,pos_bigrams_train,labels_train)

#Load Lexicons

#Socal Lexicon
socal = SocalLexicon.SocalLexicon()
#Minqing Hu Lexicon
minqinghu = MinqingHuLexicon.MinqingHuLexicon()
#Afinn Lexicon
afinn = Afinn()
#NRC Lexicon - 5 different versions
nrc1 = NRCLexicon.NRCLexicon(0)
nrc2 = NRCLexicon.NRCLexicon(1)
nrc3 = NRCLexicon.NRCLexicon(2)
nrc4 = NRCLexicon.NRCLexicon(3)
nrc5 = NRCLexicon.NRCLexicon(4)
#MPQA Lexicon
mpqa = MPQALexicon.MPQALexicon()

lexicons = [socal,minqinghu,afinn,nrc1,nrc2,nrc3,nrc4,nrc5,mpqa]

#assign a precision and F1 score to each word of a lexicon
lexicon_precision_negative, lexicon_f1_negative, lexicon_precision_positive, lexicon_f1_positive  = getLexiconF1andPrecision(mpqa, messages_train, labels_train)

#get negations list
negationList = negations.loadNegations();

#load word clusters
clusters = Clusters.Clusters()

#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary,lexicons,pos_bigrams_train,pos_bigrams_scores_negative,pos_bigrams_scores_positive,lexicon_precision_negative, lexicon_f1_negative, lexicon_precision_positive, lexicon_f1_positive, negationList,clusters)
#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary,lexicons,pos_bigrams_test,pos_bigrams_scores_negative,pos_bigrams_scores_positive,lexicon_precision_negative, lexicon_f1_negative, lexicon_precision_positive, lexicon_f1_positive, negationList,clusters)

#train classifier and return trained model
model = LogisticRegression.train(features_train,labels_train)
#model = SVM.train(features_train,labels_train)

#predict labels
prediction = LogisticRegression.predict(features_test,model)
#prediction = SVM.predict(features_test,model)
baseline_prediction= MajorityClassifier.predictPol(features_test)

#logistic regression evaluation
print "Average F1 : " +str(measures.avgF1(labels_test,prediction))
print "Baseline AverageF1 : " +str(measures.avgF1(labels_test,baseline_prediction))
print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
print "Baseline Accuracy : "+str(measures.accuracy(labels_test,baseline_prediction))
print "F1 negative : " +str(measures.F1(labels_test,prediction,0))
print "F1 positive : " +str(measures.F1(labels_test,prediction,1))
print "Precision negative: " +str(measures.precision(labels_test,prediction,0))
print "Precision positive: " +str(measures.precision(labels_test,prediction,1))
print "Recall negative : " +str(measures.recall(labels_test,prediction,0))
print "Recall positive : " +str(measures.recall(labels_test,prediction,1))

#selection.feature_selection(features_train,labels_train)

#plot learning curve
#learningCurves.plot_learning_curve(len(messages_train),features_train,labels_train,features_test,labels_test)
#learningCurves.plot_recall_precision(len(messages_train),features_train,labels_train,features_test,labels_test)
print("--- %s seconds ---" % (time.time() - start_time))

##if __name__ == "__main__":
##    main() 
    

