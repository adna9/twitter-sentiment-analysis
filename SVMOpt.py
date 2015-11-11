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
import learningCurves
import regularization

def custom_optimizer(features_train,labels_train,features_test,labels_test):
    # test classifier for different values

    C = [x/float(1000) for x in range(1,1000)]
    scores = []

    for c in C:
        print str(C.index(c)*100/float(len(C)))+"%"

        model = SVM.train(features_train,labels_train,g=1,c=c,k="linear")
        prediction = SVM.predict(features_test,model)
        score = measures.avgF1(labels_test,prediction,0,1)
        scores.append(score)

    bestScore = max(scores)
    bestC = C[scores.index(bestScore)]

    print "best C = "+str(bestC)+" , avgF1 = "+str(bestScore)

    plt.plot(C,scores,color="blue",linewidth="2.0",label="avgF1")
    plt.ylabel('C')
    plt.xlabel("average F1")
    plt.ylim(0,1)
    plt.legend(loc="best")
    plt.show()

def optunity_optimizer(search,f):    
    #optimal_configuration, info, _ = optunity.maximize_structured(performance,search_space=search,num_evals=5)
    optimal_configuration, info, _ = optunity.maximize_structured(f,search_space=search,num_evals=50)

    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print('Solution\n========')
    print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items()))) 
              
         

def train_svm(data, labels,C):
    #model = svm.LinearSVC(C=C)
    model = svm.SVC(C=C,kernel="linear")
    model.fit(data, labels)
    return model

#@optunity.cross_validated(x=features_train, y=labels_train, num_folds=10)
def performance(x_train, y_train, x_test, y_test, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):
    
    #train model
    model = train_svm(x_train, y_train, C)
    # predict the test set
    predictions = model.decision_function(x_test)
   
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return measures.avgF1(y_test,predictions,0,1)

#phase
subjectivity=True

dataset_train = "datasets/training-set-sample.tsv"
#dataset_train = "datasets/train15.tsv"
#dataset_train = "datasets/tweets#2013.tsv"

dataset_test = "datasets/testing-set-sample.tsv"
#dataset_test = "datasets/dev15.tsv"
#dataset_test = "datasets/devtweets2013.tsv"

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

    #regularize train features to [0,1]
    #features_train=regularization.regularize(features_train)

    #get features from test messages 
    features_test = features_subjectivity.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

    #regularize test features to [0,1]
    #features_test=regularization.regularize(features_test)
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

    #regularize train features to [0,1]
    #features_train=regularization.regularize(features_train)

    #get features from test messages 
    features_test = features_polarity.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_trigrams_test,pos_bigrams_scores_negative,pos_bigrams_scores_positive,pos_trigrams_scores_negative,pos_trigrams_scores_positive,pos_tags_scores_negative,pos_tags_scores_positive,mpqaScores,negationList,clusters)

    #regularize test features to [0,1]
    #features_test=regularization.regularize(features_test)

#optunity
search = {'kernel': {'linear': {'C': [0, 1]}
                }
           }

#run decoratorn "cross_validated" in preformance method
#decorator = optunity.cross_validated(x=features_train, y=labels_train, num_folds=10)
#f = decorator(performance)
#optunity_optimizer(search,f)

#custom optimizer
#custom_optimizer(features_train,labels_train,features_test,labels_test)


#plot learning curve
learningCurves.plot_learning_curve(features_train,labels_train,features_test,labels_test)
