from sklearn.preprocessing import MinMaxScaler

def train_test_x_normalization(train_x,valid_x,test_x):
    '''

    Args:
        train_x: training set
        valid_x:validation set
        test_x:test set
    '''
    
    scaler = MinMaxScaler()
    
    normalized_train_x = scaler.fit_transform(train_x)
    normalized_valid_x = scaler.transform(valid_x)
    normalized_test_x = scaler.transform(test_x)
    
    return scaler,normalized_train_x,normalized_valid_x,normalized_test_x