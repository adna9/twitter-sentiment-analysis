from nltk import bigrams
from nltk import trigrams
from tokenizers import twokenize
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import numpy as np
import optunity
import optunity.metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def subList(pos_tags,labels,c):
    sub=[]
    for i in range(0,len(pos_tags)):
        if labels[i]==c:
            sub.append(pos_tags[i])

    return sub

def polaritySubList(subjList,labels):
    polList=[]
    for i in range(0,len(subjList)):
        if labels[i]!="neutral":
            polList.append(subjList[i])
            
    return polList

def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens
  
def getLexiconF1andPrecision(l, messages, labels):
    #initialize dictionaries (exactly the same for positive-negative messages)
    precision_obj = {}
    f1_obj = {}
    precision_sub = {}
    f1_sub = {}

    #get all words from lexicon
    words = l.lexicon

    #number of messages that are objective
    x1 = len([x for x in labels if x==0])
    #number of messages that are subjective
    x2 = len([x for x in labels if x==1])

    for word in words:
        #number of messages that contain "word" and are objective
        x3 = 0
        #number of messages that contain "word" and are subjective
        x4 = 0
        #number of messages that contain the "word"
        x5 = 0

        for i in range(0,len(messages)):
            if (word in messages[i]):
                x5+=1

                if(labels[i]==0):
                    x3+=1
                else:
                    x4+=1

        #precision
        if x5!=0:
            precision_obj[word] = x3/float(x5)
            precision_sub[word] = x4/float(x5)
        else:
            precision_obj[word] = 0
            precision_sub[word] = 0

        #recall
        if x1==0:
            recall_obj = 0
        else:
            recall_obj = x3/float(x1)
            
        if x2==0:
            recall_sub = 0
        else:
            recall_sub = x4/float(x2)

        #F1
        if (precision_obj[word] + recall_obj)==0:
            f1_obj[word] = 0
        else:
            f1_obj[word] = (2*precision_obj[word]*recall_obj)/float(precision_obj[word] + recall_obj)

        if (precision_sub[word] + recall_sub)==0:
            f1_sub[word] = 0
        else:
            f1_sub[word] = (2*precision_sub[word]*recall_sub)/float(precision_sub[word] + recall_sub)
            

    return precision_obj, f1_obj, precision_sub, f1_sub

#calculate F1 and Precision scores for every word of every lexicon
def getScores(lexicons,messages, labels):
    scores = []
    for lexicon in lexicons:
        x1, x2, x3, x4 = getLexiconF1andPrecision(lexicon, messages, labels)
        scores.append(x1)
        scores.append(x2)
        scores.append(x3)
        scores.append(x4)

    return scores

def getPosTagsSet(pos_tags):
    s = set()
    
    for x in pos_tags:
        for pos_tag in x:
            s.add(pos_tag)


    return list(s)
        
    
def getBigramsSet(pos_bigrams):
    s = set()
    
    for x in pos_bigrams:
        for bigram in x:
            s.add(bigram)


    return list(s)

def getTrigramsSet(pos_bigrams):
    s = set()
    
    for x in pos_bigrams:
        for bigram in x:
            s.add(bigram)


    return list(s)


#calculate bigrams of every item of the list l
def getBigrams(l):
    b = []
    for x in l:
        b.append(list(bigrams(x)))

    return b

#calculate trigrams of every item of the list l
def getTrigrams(l):
    tr = []
    for x in l:
        tr.append(list(trigrams(x)))

    return tr


def posTagsScore(postags,category,pos_tags,labels):
    
    #keep pos tagsof specific category
    pos_tags_category = subList(pos_tags,labels,category)

    #initialize dictionary
    d = {}

    #calculate score for every bigram
    for postag in postags:
        d[postag] = score(postag,category,pos_tags_category,pos_tags)


    return d


def posBigramsScore(bigrams,category,pos_tags_bigrams,labels):
    #keep pos tags bigrams of specific category
    bigrams_category = subList(pos_tags_bigrams,labels,category)

    #initialize dictionary
    d = {}

    #calculate score for every bigram
    for bigram in bigrams:
        d[bigram] = score(bigram,category,bigrams_category,pos_tags_bigrams)


    return d

def posTrigramsScore(trigrams,category,pos_tags_trigrams,labels):
    
    #keep pos tags bigrams of specific category
    trigrams_category = subList(pos_tags_trigrams,labels,category)

    #initialize dictionary
    d = {}

    #calculate score for every bigram
    for trigram in trigrams:
        d[trigram] = score(trigram,category,trigrams_category,pos_tags_trigrams)

    return d

def score(bigram,category,bigrams_category,pos_tags_bigrams):
    #messages of "category" containing "bigram"
    x1 = 0
    for i in range(0,len(bigrams_category)):
        if bigram in bigrams_category[i]:
            x1+=1

    #messages containing "bigram"
    x2 = 0
    for i in range(0,len(pos_tags_bigrams)):
        if bigram in pos_tags_bigrams[i]:
            x2 += 1

    #messages of "category"
    x3 = len(bigrams_category)

    if(x2==0):
        precision=0
    else:
        precision = x1/float(x2)
        
    recall = x1/float(x3)
    

    #return f1 score
    if precision==0 or recall==0:
        return 0
    
    return (2*precision*recall)/float(precision + recall)

def custom_optimizer(features_train,labels_train,features_test,labels_test,C):
    # test classifier for different values

    #C = [x/float(1000) for x in range(1,1000)]
    scores = []

    for c in C:
        #print str(C.index(c)*100/float(len(C)))+"%"
        s=0

        #3 evals
        for x in range(0,3):
            model = SVM.train(features_train,labels_train,g=1,c=c,k="linear")
            prediction = SVM.predict(features_test,model)
            score = measures.avgF1(labels_test,prediction,0,1)
            s+=score
            
        s = s/float(3)
        scores.append(s)
        
        print "c = "+str(c)+" score = "+str(score)

    bestScore = max(scores)
    bestC = C[scores.index(bestScore)]

    print "best C = "+str(bestC)+" , avgF1 = "+str(bestScore)

##    plt.plot(C,scores,color="blue",linewidth="2.0",label="avgF1")
##    plt.ylabel('C')
##    plt.xlabel("average F1")
##    plt.ylim(0,1)
##    plt.legend(loc="best")
##    plt.show()

def sklearn_optimizer(C,features_train,labels_train,features_test,labels_test):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.25, random_state=0)

    tuned_parameters = [{'C': C}]

    clf = GridSearchCV(svm.LinearSVC(), tuned_parameters, cv=5,scoring=custom_scorer)
    clf.fit(X_train, y_train)

    print("Grid scores on development set:")
    print(" ")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print(" ")
    print("Best parameters set found on development set:")
    print(" ")
    print(clf.best_params_)
    print(" ")
    print("Detailed classification report:")
    print(" ")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print(" ")
    y_true, y_pred = y_test, clf.predict(X_test)
    #print(classification_report(y_true, y_pred))
    print(measures.avgF1(np.array(y_true),y_pred,0,1))
    print(" ")
    print("FINAL:")
    print(" ")
    bestC = clf.best_params_["C"]
    model = SVM.train(features_train,labels_train,c=bestC,k="linear")
    prediction = SVM.predict(features_test,model)
    print(measures.avgF1(labels_test,prediction,0,1))
    print(" ")
    

#calculates average f1 score
def custom_scorer(estimator, X, y):
	prediction = estimator.predict(X)
	return measures.avgF1(y,prediction,0,1)


def optunity_optimizer(search,f):    
    #optimal_configuration, info, _ = optunity.maximize_structured(performance,search_space=search,num_evals=5)
    optimal_configuration, info, _ = optunity.maximize_structured(f,search_space=search,num_evals=50)

    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print('Solution\n========')
    print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))

    return solution["C"]
              
def train_svm(data, labels,C):
    #model = svm.LinearSVC(C=C)
    #print C
    #model = svm.SVC(C=C,kernel="linear",cache_size=2000)
    #model = linear_model.SGDClassifier()
    #model.fit(data, labels)
    model = SVM.train(data,labels,c=C,k="linear")
    return model

#@optunity.cross_validated(x=features_train, y=labels_train, num_folds=10)
def performance(x_train, y_train, x_test, y_test, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):

    #train model
    model = train_svm(x_train, y_train, C)
    # predict the test set
    #predictions = model.decision_function(x_test)
    predictions = SVM.predict(x_test,model)
   
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return measures.avgF1(y_test,predictions,0,1)
