def accuracy(label,prediction):
    errors=0
    for i in range(0,len(label)) :
        if(label[i]!=prediction[i]):
            errors+=1

    return (len(label)-errors)/float(len(label))

def error(label,prediction):
    return 1-accuracy(label,prediction)

def avgF1(label,prediction,c1,c2):
    return (F1(label,prediction,c1) + F1(label,prediction,c2))/2

def F1(label,prediction,category):
    pr = precision(label,prediction,category)
    r = recall(label,prediction,category)
    
    return (0 if (pr==0 and r==0) else (2*pr*r)/float(pr+r))
    #return (2*pr*r)/float(pr+r)

def precision(label,prediction,category):
    #the number of messages that belong in C and were classified as C
    truePositives = 0
    
    for i in range(0,len(label)):
        if category==label[i] and label[i]==prediction[i]:
            truePositives+=1

    #the number of messages that were classified as C
    y = len([c for c in prediction if c==category])

    if(y==0) :
        return 0

    return truePositives/float(y)

def recall(label,prediction,category):
    
    #the number of messages that belong in C and were classified as C
    truePositives = 0
    for i in range(0,len(label)):
        if category==label[i] and label[i]==prediction[i]:
            truePositives+=1

    #the number of messages that belong in C
    y = len([c for c in label if c==category])

    if(y==0) :
        return 0
    
    return truePositives/float(y)

