#Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

#train model
def train(features,labels):
    #define classifier
    model = LogisticRegression(C=1e5)

    #fit the data
    model.fit(features, labels)

    return model

#predict labels
#default threshold=0.5
def predict(features,model,threshold=0.5):

    labels=[1 if posprob>threshold else 0 for negprob,posprob in model.predict_proba(features) ]
    return labels

