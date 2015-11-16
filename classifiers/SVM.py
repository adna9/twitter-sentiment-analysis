#SVM Classifier

from sklearn import svm

#train model
def train(features,labels,g=0,c=1,k="rbf",coef0=0,degree=2):
    #define classifier
    if k=="linear":
        model = svm.LinearSVC(C=c)
        #model = svm.SVC(C=c,kernel=k)
    elif k=="poly":
        model=svm.SVC(C=c,kernel=k,degree=degree,coef0=coef0)
    else:
        model=svm.SVC(C=c,kernel=k,gamma=g)

    #fit data
    model.fit(features,labels)

    return model

#predicts labels
def predict(features,model):
    return model.predict(features)




