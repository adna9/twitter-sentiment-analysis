from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def feature_selection(features_train,labels_train,features_test,K):
    fs = SelectKBest(f_classif,K)
    fs.fit(features_train,labels_train)

    features_train_new = fs.transform(features_train)
    features_test_new = fs.transform(features_test)
    
    return features_train_new,features_test_new


