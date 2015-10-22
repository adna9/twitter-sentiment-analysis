from nltk import bigrams
from tokenizers import twokenize

def subList(pos_tags,labels,c):
    sub=[]
    for i in range(0,len(pos_tags)):
        if labels[i]==c:
            sub.append(pos_tags[i])

    return sub
def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens

def getLexiconF1andPrecision(lexicon, messages, labels):
    #initialize dictionaries
    precision_obj = {}
    f1_obj = {}
    precision_sub = {}
    f1_sub = {}

    #get all words from lexicon
    words = lexicon.d.keys()

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
def getBigramsSet(pos_bigrams):
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

def posBigramsScore(bigrams,category,pos_tags_bigrams,labels):
    #keep pos tags bigrams of specific category
    bigrams_category = subList(pos_tags_bigrams,labels,category)

    #initialize dictionary
    d = {}

    #calculate score for every bigram
    for bigram in bigrams:
        d[bigram] = score(bigram,category,bigrams_category,pos_tags_bigrams)


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
