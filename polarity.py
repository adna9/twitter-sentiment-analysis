from features import features_polarity as features
from classifiers import LogisticRegression, SVM, MajorityClassifier
import learningCurves
from utilities import *
from feature_selection import selection
import regularization
from evaluation import measures

def classify(messages_train,labels_train,messages_test,process_messages_train,process_messages_test,tokens_train,tokens_test,process_tokens_train,process_tokens_test,pos_tags_train,pos_tags_test,negationList,clusters,slangDictionary,lexicons,mpqa_lexicons): 
    # 0 - negative messages
    # 1 - positives messages
    labels_train = [0 if x=="negative" else 1 for x in labels_train]
    
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
    features_train = features.getFeatures(messages_train,process_messages_train,tokens_train,process_tokens_train,pos_tags_train,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_train,pos_trigrams_train,pos_bigrams_scores_negative,pos_bigrams_scores_positive,pos_trigrams_scores_negative,pos_trigrams_scores_positive,pos_tags_scores_negative,pos_tags_scores_positive,mpqaScores,negationList,clusters)

    
    #regularize train features
    #features_train=regularization.regularize(features_train)


    #get features from test messages 
    features_test = features.getFeatures(messages_test,process_messages_test,tokens_test,process_tokens_test,pos_tags_test,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams_test,pos_trigrams_test,pos_bigrams_scores_negative,pos_bigrams_scores_positive,pos_trigrams_scores_negative,pos_trigrams_scores_positive,pos_tags_scores_negative,pos_tags_scores_positive,mpqaScores,negationList,clusters)


    #regularize test features
    #features_test=regularization.regularize(features_test)

        
    #train classifier and return trained model
    #model = LogisticRegression.train(features_train,labels_train)
    model = SVM.train(features_train,labels_train,g=0,c=0.17578125,k="linear",coef0=0,degree=2)
    #model = SVM.train(features_train,labels_train)

        
    #predict labels
    #prediction = LogisticRegression.predict(features_test,model)
    prediction = SVM.predict(features_test,model)

    return prediction

def evaluate(prediction,labels_test):
    
    labels_test = [0 if x=="neutral" else 1 if x=="positive" else -1 for x in labels_test]

    #logistic regression evaluation
    print "Average F1 : " +str(measures.avgF1(labels_test,prediction,-1,1))
    #print "Baseline AverageF1 : " +str(measures.avgF1(labels_test,baseline_prediction))
    print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
    #print "Baseline Accuracy : "+str(measures.accuracy(labels_test,baseline_prediction))
    print "F1 negative : " +str(measures.F1(labels_test,prediction,-1))
    print "F1 positive : " +str(measures.F1(labels_test,prediction,1))
    print "Precision negative: " +str(measures.precision(labels_test,prediction,-1))
    print "Precision positive: " +str(measures.precision(labels_test,prediction,1))
    print "Recall negative : " +str(measures.recall(labels_test,prediction,-1))
    print "Recall positive : " +str(measures.recall(labels_test,prediction,1))
