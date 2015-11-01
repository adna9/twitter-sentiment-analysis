from features import features_subjectivity as features
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
from utilities import *
import learningCurves
import regularization
from preProcess import *
from postaggers import arktagger
from evaluation import measures

def classify(messages_train,labels_train,messages_test,negationList,clusters,slangDictionary,lexicons,mpqa_lexicons,dictionary):

    #merge positive-negative categories into one category(subjective), as we
    #want to check the subjectivity of message
    # 0 - objective(neutral) messages
    # 1 - subjective(positive or negatve) messages
    labels_train = [0 if x=="neutral" else 1 for x in labels_train]

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
    features_train = features.getFeatures(messages_train,process_messages_train,tokens_train,process_tokens_train,pos_tags_train,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_train,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

    #regularize train features to [0,1]
    features_train=regularization.regularize(features_train)
  
    #get features from test messages 
    features_test = features.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,mpqaScores,negationList,clusters)

    #regularize test features to [0,1]
    features_test=regularization.regularize(features_test)

    #train classifier and return trained model
    model = LogisticRegression.train(features_train,labels_train)
    #model = SVM.train(features_train,labels_train)

    #predict labels
    prediction = LogisticRegression.predict(features_test,model)
    #prediction = SVM.predict(features_test,model)

    return messages_train,messages_test,process_messages_train,process_messages_test,tokens_train,tokens_test,process_tokens_train,process_tokens_test,pos_tags_train,pos_tags_test,prediction

def evaluate(prediction,labels_test):

    labels_test = [0 if x=="neutral" else 1 for x in labels_test]

    #logistic regression evaluation
    print "Average F1 : " +str(measures.avgF1(labels_test,prediction,0,1))
    #print "Baseline AverageF1 : " +str(measures.avgF1(labels_test,baseline_prediction))
    print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
    #print "Baseline Accuracy : "+str(measures.accuracy(labels_test,baseline_prediction))
    print "F1 objective : " +str(measures.F1(labels_test,prediction,0))
    print "F1 subjective : " +str(measures.F1(labels_test,prediction,1))
    print "Precision objective: " +str(measures.precision(labels_test,prediction,0))
    print "Precision subjective: " +str(measures.precision(labels_test,prediction,1))
    print "Recall objective : " +str(measures.recall(labels_test,prediction,0))
    print "Recall subjective : " +str(measures.recall(labels_test,prediction,1))

