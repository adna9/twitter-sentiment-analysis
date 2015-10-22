

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def feature_selection(features,labels):
    print(features.shape)

    features_new = SelectKBest(f_classif,50).fit_transform(features,labels)

    print(features_new.shape)


