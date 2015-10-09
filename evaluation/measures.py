def accuracy(A,B):
    #if A[i]==B[i] then A[i]+B[i]==0 or A[i]+B[i]==2
    #else if A[i]/=B[i] the A[i]+B[i]=1
    errors = sum((A+B)==1)

    return (len(A)-errors)/float(len(A))

def error(A,B):
    return 1-accuracy(A,B)

def avgF1(A,B):
    return (F1(A,B,0) + F1(A,B,1))/2

def F1(A,B,category):
    pr = precision(A,B,category)
    r = recall(A,B,category)

    return (2*pr*r)/float(pr+r)

def precision(A,B,category):
    #the number of messages that belong in C and were classified as C
    x = 0
    for i in range(0,len(A)):
        if category==A[i] and A[i]==B[i]:
            x+=1

    #the number of messages that were classified as C
    y = len([c for c in B if c==category])

    return x/float(y)

def recall(A,B,category):
    #the number of messages that belong in C and were classified as C
    x = 0
    for i in range(0,len(A)):
        if category==A[i] and A[i]==B[i]:
            x+=1

    #the number of messages that beling in C
    y = len([c for c in A if c==category])

    return x/float(y)


