from sklearn.neighbors import KNeighborsClassifier

#train model
def train(features,labels):
    
    model = KNeighborsClassifier(n_neighbors=8)
    
    model.fit(features,labels)
    
    return model

#predict labels
def predict(features,model):
    return model.predict(features)
