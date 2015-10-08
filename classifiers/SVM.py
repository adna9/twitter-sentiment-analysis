#SVM Classifier

from sklearn import svm

#train model
def train(features,labels):
    #define classifier
    model=svm.SVC(gamma=0.0001,C=100)

    #fit data
    model.fit(features,labels)

    return model

#predicts labels
def predict(features,model):
    return model.predict(features)




