import polarity
import subjectivity
from tsvfiles import tsvreader
import time
from clusters import Clusters
from lexicons import negations,Slang,SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
import enchant
from utilities import polaritySubList
from preProcess import *
from postaggers import arktagger
from utilities import *
from evaluation import measures
from features import features_subjectivity,features_polarity
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
import optunity
import optunity.metrics
from sklearn import svm
from sklearn import linear_model
import learningCurves
import regularization
import time
from feature_selection import selection

#phase
subjectivity = True
feature_selection = False

#dataset_train = "datasets/training-set-sample.tsv"
dataset_train = "datasets/train15.tsv"
#dataset_train = "datasets/tweets#2013.tsv"
#dataset_train = "datasets/full_train.tsv"

#dataset_test = "datasets/testing-set-sample.tsv"
dataset_test = "datasets/dev15.tsv"
#dataset_test = "datasets/devtweets2013.tsv"
#dataset_test = "datasets/full_dev.tsv"

if subjectivity:
    #load training set
    labels_train,messages_train=tsvreader.opentsv(dataset_train)

    #load testing set
    labels_test, messages_test = tsvreader.opentsv(dataset_test)
else:
    #load training set
    labels_train,messages_train=tsvreader.opentsvPolarity(dataset_train)

    #load testing set
    labels_test, messages_test = tsvreader.opentsvPolarity(dataset_test)

#get negations list
negationList = negations.loadNegations();

#load word clusters
clusters = Clusters.Clusters()

#load Slang Dictionary
slangDictionary = Slang.Slang()

#load general purpose dictionary
dictionary = enchant.Dict("en_US")

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages (after preprocessing the messages pos tags will be recomputed)
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)

#preprocess messages
process_messages_train = preprocessMessages(messages_train,tokens_train,pos_tags_train,slangDictionary,dictionary)
process_messages_test = preprocessMessages(messages_test,tokens_test,pos_tags_test,slangDictionary,dictionary)

#tokenize process messages (final pos tags)
process_tokens_train=tokenize(process_messages_train)
process_tokens_test=tokenize(process_messages_test)

#compute pos tags for all preprocessed messages
pos_tags_train = arktagger.pos_tag_list(process_messages_train)
pos_tags_test = arktagger.pos_tag_list(process_messages_test)

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

#SEMEVAL_13 Lexicons
semval_neutral = MPQALexicon.MPQALexicon(8)
semval_positive = MPQALexicon.MPQALexicon(9)
semval_negative = MPQALexicon.MPQALexicon(10)

#MPQA + SEMEVAL_13 Lexicons
mpqa_lexicons = [S_pos,S_neg,S_pos_neg,S_neu,W_pos,W_neg,W_pos_neg,W_neu,semval_neutral,semval_positive,semval_negative]

if subjectivity:
    #merge positive-negative categories into one category(subjective), as we
    #want to check the subjectivity of message
    # 0 - objective(neutral) messages
    # 1 - subjective(positive or negatve) messages
    labels_train = [0 if x=="neutral" else 1 for x in labels_train]
    labels_test = [0 if x=="neutral" else 1 for x in labels_test]

    #compute pos tag bigrams for all messages
    pos_bigrams_train = getBigrams(pos_tags_train)
    pos_bigrams_test = getBigrams(pos_tags_test)

    #get the unique pos bigrams from training set
    unique_bigrams = getBigramsSet(pos_bigrams_train)

    #calculate pos bigrams score for all categories
    #both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
    pos_bigrams_scores_objective = posBigramsScore(unique_bigrams,0,pos_bigrams_train,labels_train)
    pos_bigrams_scores_subjective = posBigramsScore(unique_bigrams,1,pos_bigrams_train,labels_train)

    #assign a precision and F1 score to each word of to all mpqa and semeval_13 lexicons
    mpqaScores = getScores(mpqa_lexicons,process_messages_train,labels_train)

    #get features from train messages
    features_train = features_subjectivity.getFeatures(messages_train,process_messages_train,tokens_train,process_tokens_train,pos_tags_train,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_train,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

    #regularize train features
    features_train=regularization.regularize(features_train)

    #get features from test messages 
    features_test = features_subjectivity.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

    #regularize test features
    features_test=regularization.regularize(features_test)
else:
    # 0 - negative messages
    # 1 - positives messages
    labels_train = [0 if x=="negative" else 1 for x in labels_train]
    labels_test = [0 if x=="negative" else 1 for x in labels_test]
    
    #compute pos tag bigrams for all messages
    pos_bigrams_train = getBigrams(pos_tags_train)
    pos_bigrams_test = getBigrams(pos_tags_test)

    #compute pos tag trigrams for all messages
    pos_trigrams_train = getTrigrams(pos_tags_train)
    pos_trigrams_test = getTrigrams(pos_tags_test)

    #get the unique pos bigrams and trigrams from training set
    unique_pos_tags = getPosTagsSet(pos_tags_train)
    unique_bigrams = getBigramsSet(pos_bigrams_train)
    unique_trigrams= getTrigramsSet(pos_trigrams_train)

    #calculate pos bigrams score for all categories
    #both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
    pos_tags_scores_negative = posTagsScore(unique_pos_tags,0,pos_tags_train,labels_train)
    pos_tags_scores_positive = posTagsScore(unique_pos_tags,1,pos_tags_train,labels_train)

    #calculate pos bigrams score for all categories
    #both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
    pos_bigrams_scores_negative = posBigramsScore(unique_bigrams,0,pos_bigrams_train,labels_train)
    pos_bigrams_scores_positive = posBigramsScore(unique_bigrams,1,pos_bigrams_train,labels_train)

    #calculate pos bigrams score for all categories
    #both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
    pos_trigrams_scores_negative = posTrigramsScore(unique_trigrams,0,pos_trigrams_train,labels_train)
    pos_trigrams_scores_positive = posTrigramsScore(unique_trigrams,1,pos_trigrams_train,labels_train)

    #assign a precision and F1 score to each word of to all mpqa lexicons
    mpqaScores = getScores(mpqa_lexicons,process_messages_train,labels_train)

    #get features from train messages
    features_train = features_polarity.getFeatures(messages_train,process_messages_train,tokens_train,process_tokens_train,pos_tags_train,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_train,pos_trigrams_train,pos_bigrams_scores_negative,pos_bigrams_scores_positive,pos_trigrams_scores_negative,pos_trigrams_scores_positive,pos_tags_scores_negative,pos_tags_scores_positive,mpqaScores,negationList,clusters)

    #regularize train features
    features_train=regularization.regularize(features_train)

    #get features from test messages 
    features_test = features_polarity.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_trigrams_test,pos_bigrams_scores_negative,pos_bigrams_scores_positive,pos_trigrams_scores_negative,pos_trigrams_scores_positive,pos_tags_scores_negative,pos_tags_scores_positive,mpqaScores,negationList,clusters)

    #regularize test features
    features_test=regularization.regularize(features_test)



t1 = time.time()

C=[1,2,3,4,5]
sklearn_optimizer(C,features_train,labels_train,features_test,labels_test)

#optunity
##search = {'kernel': {'linear': {'C': [0, 32]}
##                }
##           }

#run decoratorn "cross_validated" in preformance method
##decorator = optunity.cross_validated(x=features_train, y=labels_train, num_folds=10)
##f = decorator(performance)
##optunity_optimizer(search,f)

#optunity
##search2 = {'kernel': {'linear': {'C': [1, 10]}
##                }
##           }
##
###run decoratorn "cross_validated" in preformance method
##decorator = optunity.cross_validated(x=features_train, y=labels_train, num_folds=10)
##f = decorator(performance)
##optunity_optimizer(search2,f)


#custom optimizer
#C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#C = [x * 0.01 for x in range(0, 1000)]
#C.remove(0)
#custom_optimizer(features_train,labels_train,features_test,labels_test,C)


#plot learning curve
#learningCurves.plot_learning_curve(features_train,labels_train,features_test,labels_test)

##if feature_selection:
##    search = {'kernel': {'linear': {'C': [0, 1]}
##                }
##           }
##    
##    K = list(np.arange(50,features_train.shape[1],50))
##    if K[len(K)-1]!=features_train.shape[1]:
##        K.append(features_train.shape[1])
##
##    C = []
##    scores = []
##
##    for k in K:
##        #select k best features
##        features_train_new, features_test_new = selection.feature_selection(features_train,labels_train,features_test,k)
##
##        #tune SVM with new training set
##        decorator = optunity.cross_validated(x=features_train_new, y=labels_train, num_folds=10)
##        f = decorator(performance)
##        c = optunity_optimizer(search,f)
##
##        C.append(c)
##
##        #calculate score
##        model = SVM.train(features_train_new,labels_train,c=c,k="linear")
##        predictions = SVM.predict(features_test_new,model)
##        score = measures.avgF1(labels_test,predictions,0,1)
##        scores.append(score)
##
##        print "k="+str(k)+" C="+str(c)+" score="+str(score)
##


t2 = time.time()
print "total time : "+str(t2-t1)
