def accuracy(label,prediction):
    #if A[i]==B[i] then A[i]+B[i]==0 or A[i]+B[i]==2
    #else if A[i]/=B[i] the A[i]+B[i]=1
    errors = sum((label+prediction)==1)

    return (len(label)-errors)/float(len(label))

def error(label,prediction):
    return 1-accuracy(label,prediction)

def avgF1(label,prediction):
    return (F1(label,prediction,0) + F1(label,prediction,1))/2

def F1(label,prediction,category):
    pr = precision(label,prediction,category)
    r = recall(label,prediction,category)

    return (2*pr*r)/float(pr+r)

def precision(label,prediction,category):
    #the number of messages that belong in C and were classified as C
    truePositives = 0
    
    for i in range(0,len(label)):
        if category==label[i] and label[i]==prediction[i]:
            truePositives+=1

    #the number of messages that were classified as C
    y = len([c for c in prediction if c==category])

    if(y==0) :
        return -1

    return truePositives/float(y)

def recall(label,prediction,category):
    
    #the number of messages that belong in C and were classified as C
    truePositives = 0
    for i in range(0,len(label)):
        if category==label[i] and label[i]==prediction[i]:
            truePositives+=1

    #the number of messages that beling in C
    y = len([c for c in label if c==category])

    if(y==0) :
        return -1
    
    return truePositives/float(y)

