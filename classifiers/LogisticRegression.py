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
def predict(features,model):
    return model.predict(features)
