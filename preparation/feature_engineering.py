import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler


############################################################################

# Feature engineering

############################################################################


def select_kbest(x, y, k):
    '''
    
    Description:
    -----------
    This function takes in a dataframe, a target, and a number that is less than or equal to total number of features. 
    The dataframe is split and scaled, features are separated into objects and numberical columns, and 
    finally the Select KBest test is run and returned.
    
    Parameters:
    ----------
    x: dataframe
        The dataframe being explored
    y: str
        String that is the target feature
    k: int
        Number of features to return
        
    '''
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(x, y)
    object_cols = get_object_cols(x)
    numeric_cols = get_numeric_X_cols(X_train, object_cols)
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X_train_scaled, y_train)
    feature_mask = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    return f_feature


def rfe(x, y, k):
    '''
    
    Description:
    -----------
    This function takes in a dataframe, a target, and a number that is less than or equal to total number of features. 
    The dataframe is split and scaled, the features are separated into objects and numberical columns, and 
    finally the RFE test is run and returned.

    Parameters:
    ----------
    x: dataframe
        The dataframe being explored
    y: str
        String that is the target feature
    k: int
        Number of features to return
        
    '''
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(x, y)
    object_cols = get_object_cols(x)
    numeric_cols = get_numeric_X_cols(X_train, object_cols)
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    lm = LinearRegression()
    rfe = RFE(lm, k)
    rfe.fit(X_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    return rfe_feature

