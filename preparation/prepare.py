import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler


############################################################################

# Split, Scale Data

############################################################################


def split_dataframe(df, stratify_by=None, rand=42, test_size=.2, validate_size=.3):
    """
    
    Description:
    -----------
    Utility function to create train, validate, and test splits.
    Generates train, validate, and test samples from a dataframe.
    Credit to @ryanorsinger
    
    Parameters:
    ----------
    df : DataFrame
        The dataframe to be split
    stratify_by : str
        Name of the target variable. Ensures different results of target variable are spread between the samples. Default is None.
    test_size : float
        Ratio of dataframe (0.0 to 1.0) that should be kept for testing sample. Default is 0.2.
    validate_size: float
        Ratio of train sample (0.0 to 1.0) that should be kept for validate sample. Default is 0.3.
    random_stat : int
        Value provided to create re-produceable results. Default is 1414.
    Returns
    -------
    DataFrame
        Three dataframes representing the training, validate, and test samples
        
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=test_size, random_state=rand)
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand)
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=rand, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand, stratify=train[stratify_by])

    return train, validate, test

#--------------------------------------------------------------------------#


def remove_outliers(df, col, multiplier=1.5):
    '''
    
    Description:
    -----------
    This function takes in a dataframe, a column name and, a multiplier as a float and returns a dataframe with outliers removed.
    
    Parameters:
    -----------
    df: dataframe
        The dataframe being explored
    col: str
        A string that contains a continuous variable (column)
    multiplier: float
        Default = 1.5, As the multiplier increases, the number of outliers removed decreases 
    
    '''
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    df = df[df[col] > lower_bound]
    df = df[df[col] < upper_bound]
    return df




#--------------------------------------------------------------------------#


def X_train_validate_test(df, target):
    '''
    
    Description:
    -----------
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    
    Parameters:
    ----------
    df: Dataframe
        Prepped and cleaned dataframe
    target: str
        Column of the target variable as a string
        
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test




def get_object_cols(df):
    '''

    Description:
    -----------
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    
    Parameters:
    ----------
    df: Dataframe
    
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")
        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols



def get_numeric_X_cols(X_train, object_cols):
    '''
    
    Description:
    -----------
    Takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    
    Parameters:
    ----------
    X_train: Dataframe
        Train dataframe in which the target variable has been removed
        
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols



def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    
    Description:
    -----------
    This function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    
    Parameters:
    ----------
    X_train: Dataframe
        Train dataframe in which the target variable has been removed
    X_validate: Dataframe
        Validate dataframe in which the target variable has been removed
    X_test: Dataframe
        Test dataframe in which the target variable has been removed
    numeric_cols: list
        List of columns with numeric datatypes
        
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled


#--------------------------------------------------------------------------#


