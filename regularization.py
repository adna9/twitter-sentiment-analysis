from sklearn import preprocessing

def regularize(features):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(features)
