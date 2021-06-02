import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns


############################################################################

# Probablistic Anomaly Detection

############################################################################



def generate_column_counts_df(df, target_col):
    """
    Generates a dataframe containing the counts of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable and the counts of the target occurring within the dataframe.
    """

    return df[target_col].value_counts(dropna=False).reset_index().\
                rename(columns={'index': target_col, target_col : target_col + '_count'})

def generate_column_probability_df(df, target_col):
    """
    Generates a dataframe containing the probability of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable and the probabilities of the target occurring within the dataframe.
    """

    return (df[target_col].value_counts(dropna=False)/df[target_col].count()).reset_index().\
                rename(columns={'index': target_col, target_col : target_col + '_proba'})

def generate_counts_and_probability_df(df, target_var):
    """
    Generates a dataframe containing the counts and probabilities of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable, the counts, and probabilities of the target occurring within the dataframe.
    """

    counts_df = generate_column_counts_df(df, target_var)
    probability_df = generate_column_probability_df(df, target_var)

    return counts_df.merge(probability_df)

def visualize_target_counts(df, target_var, target_counts, fig_size=(12, 8)):
    """
    Creates a barplot of the different values for the target variable and the count for each.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column
    target_counts : string
        The key name for the counts of the target variable column
    fig_size : tuple of (int, int)
        The dimensions for the barplot (default=(12,8))

    Returns
    -------
    None
    """

    plt.figure(figsize=fig_size)
    
    splot = sns.barplot(data=df, x = target_var, y = target_counts, ci = None)

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', xytext = (0, 10), 
                       textcoords = 'offset points'
                       )
    plt.xticks(rotation='vertical')
    plt.show()

def generate_conditional_probability_df(df, target_var, conditional_var):
    """
    Generates a dataframe containing the conditional probability of a target variable occurring given a conditional variable's presence.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target and conditional variables
    target_var : string
        The key name for the target variable column
    conditional_var : string
        The key name for the conditional variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable, the conditional variable, and the probability of the combination occurring within the dataframe.
    """

    probabilities = df.groupby([conditional_var]).size().div(len(df))

    conditional_proba_df = pd.DataFrame(df.groupby([conditional_var, target_var]).size().div(len(df)).\
                              div(probabilities, axis=0, level=conditional_var).\
                              reset_index().\
                              rename(columns={0 : 'proba_' + target_var + '_given_' + conditional_var}))

    return conditional_proba_df



############################################################################

# Time Series Anaomaly Detection

############################################################################



def bollinger_bands(target_series, span, weight, target_col_name):
    '''
    
    Description:
    ------------
    This function takes in a series, a span, a weight, and a user.
    Then creates bollinger bands, midband, upper band, and lower band, 
    then calculates the percent bandwidth of each day, finally
    returns a dataframe containing the information specific to a single user.
    
    Parameters:
    -----------
    target_series: series
        Series with a datetime index
    span: int
        Span is the number of days to be used in exponential weighted functions
    weight: int
        Weight is a value, k, that is to be multiplied to the standard deviation, and 
        is used to create the upper and lower bands.
    user: int
        The specific user to be highlighted
    target_col_name: str
        Holds the column name of the target series

    '''
    # creates midband
    midband = target_series.ewm(span=span).mean()
    
    # establishes the standard deviation
    stdev = target_series.ewm(span=span).std()
    
    # creates upper and lower bands 
    ub = midband + stdev*weight
    lb = midband - stdev*weight
    
    # combines upper and lower bands as a dataframe
    bb = pd.concat([ub, lb], axis=1)
    
    # combines the pages and midband to the bb_df, and renames the columns
    bb_df = pd.concat([target_series, midband, bb], axis=1)
    bb_df.columns = [target_col_name, 'midband', 'ub', 'lb']
    
    # creates the column that holds the percent bandwidth
    bb_df['pct_b'] = (my_df[target_col_name] - my_df['lb'])/(my_df['ub'] - my_df['lb'])
        
    # return dataframe
    return bb_df



def plt_bands(bb_df, target_col_name):
    '''
    
    Description:
    ------------
    This function plots a bollinger bands graph of a specific user.
    
    Parameters:
    -----------
    my_df: dataframe
        A dataframe containing the columns: 'pages', 'midband', 'ub', 'lb'
    target_col_name: str
        Holds the column name of the target series
    
    '''
    # creates sub plots and sets figure size
    fig, ax = plt.subplots(figsize=(12,8))
    
    # x = time, y = number of logs
    ax.plot(bb_df.index, bb_df[target_col_name], label='Number of Logs: '+str(target_col_name))
    
    # x = time, y = midband
    ax.plot(bb_df.index, bb_df.midband, label = 'EMA/midband')
    
    # x = time, y = upper band
    ax.plot(bb_df.index, bb_df.ub, label = 'Upper Band')
    
    # x = time, y = lower band
    ax.plot(bb_df.index, bb_df.lb, label = 'Lower Band')
    
    # creates the legend
    ax.legend(loc='best')
    
    # creates the y axis label
    ax.set_ylabel('Number of Logs')
    
    # displays all open figures
    plt.show()
    
def find_anomalies(df, user, span, weight):
    '''
    Description:
    ------------
    This function combines, prep, compute_pct_b, and plt_bands functions, and 
    returns a dataframe of instances where the number of logs 
    is greater than the upper band.
    
    Parameters:
    -----------
    df: dataframe
        Dataframe containing date column, user_id, and endpoint columns
    user: int
        The specific user to be highlighted
    span: int
        Span is the number of days to be used in exponential weighted functions
    weight: int
        Weight is a value, k, that is to be multiplied to the standard deviation, and 
        is used to create the upper and lower bands.
    
    '''
    # calls the prep function, returns the pages series
    pages = prep(df, user)
    
    # calls the compute_pct_b function, returns a dataframe
    bb_df = bollinger_bands(target_series, span, weight, target_col_name)
    
    # calls plt_bands function, plots a bollinger bands graph
#     plt_bands(my_df, user)   # <--- Silenced due to time
    
    # returns a dataframe of instances greater than the upper band
    return bb_df[bb_df.pct_b>1]