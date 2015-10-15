from tsvfiles import tsvreader
from features_polarity import features
from tokenizers import twokenize
from postaggers import arktagger
from nltk import bigrams
from nltk import trigrams
from lexicons import Slang
from lexicons import SocalLexicon 
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier,KNN
import learningCurves



def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens

#def main():

#load training set
dataset_train = "datasets/train15.tsv"
labels_train, messages_train = tsvreader.opentsvPolarity(dataset_train)

#load testing set
dataset_test = "datasets/dev15.tsv"
labels_test, messages_test = tsvreader.opentsvPolarity(dataset_test)

#load Slang Dictionary
slangDictionary = Slang.Slang()

#load SocalLexicon
lex =SocalLexicon.SocalLexicon()
lex.loadLexicon()

# 0 - negative messages 
# 1 - positive messages
labels_train = [0 if x=="negative" else 1 for x in labels_train]
labels_test = [0 if x=="negative" else 1 for x in labels_test]

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)

#compute bigrams for all messages
bigrams_train=[bigrams(x) for x in pos_tags_train]
bigrams_test =[bigrams(x) for x in pos_tags_test]

#compute trigrams for all messages
trigrams_train=[trigrams(x) for x in pos_tags_train]
trigrams_test =[trigrams(x) for x in pos_tags_test]

#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,bigrams_train,trigrams_train, slangDictionary,lex)

#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,bigrams_test,trigrams_test,slangDictionary,lex)

#train classifier and return trained model
model = LogisticRegression.train(features_train,labels_train)
#model = SVM.train(features_train,labels_train)           
#model = KNN.train(features_train,labels_train)

#predict labels
#prediction=SVM.predict(features_test,model)
#prediction = KNN.predict(features_test,model)
prediction= LogisticRegression.predict(features_test,model)
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

#plot learning curves

#learningCurves.plot_learning_curve(len(messages_train),features_train,labels_train,features_test,labels_test)
learningCurves.plot_recall_precision(len(messages_train),features_train,labels_train,features_test,labels_test)
learningCurves.plot_error_threshold(len(messages_train),features_train,labels_train,features_test,labels_test)


##if __name__ == "__main__":
##    main() 
    

