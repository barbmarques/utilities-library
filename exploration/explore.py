import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
import datetime as dt





##########################################################################################

# Zero's and NULLs

##########################################################################################



#----------------------------------------------------------------------------------------#
###### Identifying Zeros and Nulls in columns and rows


def missing_zero_values_table(df):
    '''
    
    Description:
    -----------
    This function takes in a dataframe and counts number of Zero values and NULL values. Returns a Table with counts and percentages of each value type.
    
    Parameters:
    ----------
    df: Dataframe
    
    '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'NULL Values', 2 : '% of Total NULL Values'})
    mz_table['Total Zero\'s plus NULL Values'] = mz_table['Zero Values'] + mz_table['NULL Values']
    mz_table['% Total Zero\'s plus NULL Values'] = 100 * mz_table['Total Zero\'s plus NULL Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
    '% of Total NULL Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str((mz_table['NULL Values'] != 0).sum()) +
          " columns that have NULL values.")
    #       mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table



def missing_columns(df):
    '''
    
    Description:
    -----------
    This function takes a dataframe, counts the number of null values in each row, and converts the information into another dataframe. Adds percent of total columns.
    
    Parameters:
    ----------
    df: Dataframe
    
    '''
    missing_cols_df = pd.Series(data=df.isnull().sum(axis = 1).value_counts().sort_index(ascending=False))
    missing_cols_df = pd.DataFrame(missing_cols_df)
    missing_cols_df = missing_cols_df.reset_index()
    missing_cols_df.columns = ['total_missing_cols','num_rows']
    missing_cols_df['percent_cols_missing'] = round(100 * missing_cols_df.total_missing_cols / df.shape[1], 2)
    missing_cols_df['percent_rows_affected'] = round(100 * missing_cols_df.num_rows / df.shape[0], 2)
    
    return missing_cols_df


#----------------------------------------------------------------------------------------#
###### Do things to the above zeros and nulls ^^

def handle_missing_values(df, drop_col_proportion, drop_row_proportion):
    '''
    
    Description:
    -----------
    This function takes in a dataframe and returns a dataframe with columns and rows that fit the input criteria removed.
    
    Parameters:
    ---------
    df: Dataframe
    drop_col_proportion: float
        a number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column, 
    drop_row_proportion: float
        a number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row, and returns the dataframe with the columns and rows dropped as indicated.
        
    '''
    # drop cols > thresh, axis = 1 == cols
    df = df.dropna(axis=1, thresh = drop_col_proportion * df.shape[0])
    # drop rows > thresh, axis = 0 == rows
    df = df.dropna(axis=0, thresh = drop_row_proportion * df.shape[1])
    return df



##########################################################################################

# Visualiation Exploration

##########################################################################################



###################### ________________________________________
### Univariate

def explore_univariate(train, categorical_vars, quant_vars):
    '''
    
    Description:
    -----------
    Takes in a dataframe and a categorical variable and returns a frequency table and barplot of the frequencies, for a given categorical variable, compute the frequency count and percent split and return a dataframe of those values along with the different classes, and takes in a dataframequantitative variable and returns descriptive stats table, histogram, and boxplot of the distributions
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    categorical_vars: list containing strings
        List of categorical variables within the train dataframe
    quant_vars: list containing strings
        List of quantitative variables within the train dataframe
        
    '''
    for cat_var in categorical_vars:
        explore_univariate_categorical(train, cat_var)
        print('_________________________________________________________________')
    for quant in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, quant)
        plt.show(p)
        print(descriptive_stats)

def explore_univariate_categorical(train, cat_var):
    '''
    
    Description:
    -----------
    Takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    cat_var: str
        A categorical variable within the train dataframe

    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant):
    '''
    
    Description:
    -----------
    Takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    quant: str
        A quantitative variable within the train dataframe
        
    '''
    descriptive_stats = train[quant].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant], color='lightseagreen')
    p = plt.title(quant)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant])
    p = plt.title(quant)
    return p, descriptive_stats
    
def freq_table(train, cat_var):
    '''
    
    Description:
    -----------
    For a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    cat_var: str
        A categorical variable within the train dataframe
        
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table

###################### ________________________________________
#### Bivariate


def explore_bivariate(train, categorical_target, continuous_target, binary_vars, quant_vars):
    '''
    
    Description:
    -----------
    This function makes use of explore_bivariate_categorical and explore_bivariate_quant functions. 
    Each of those take in a continuous target and a binned/cut version of the target to have a categorical target. 
    the categorical function takes in a binary independent variable and the quant function takes in a quantitative 
    independent variable. 
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    categorical_target: str
        The categorical target 
    continuous_target: str
        The continuous target
    binary_vars: list containing strings
        List of binary variables within the train dataframe
    quant_vars: list containing strings
        List of quantitative variables within the train dataframe
    
    '''
    for binary in binary_vars:
        explore_bivariate_categorical(train, categorical_target, continuous_target, binary)
    for quant in quant_vars:
        explore_bivariate_quant(train, categorical_target, continuous_target, quant)

###################### ________________________________________
## Bivariate Categorical

def explore_bivariate_categorical(train, categorical_target, continuous_target, binary):
    '''
    
    Description:
    -----------
    Takes in binary categorical variable and binned/categorical target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the binary categorical variable. 
    
    Parameters:
    ----------
    train: Dataframe
        Dataframe used to train models 
    categorical_target: str
        The categorical target 
    continuous_target: str
        The continuous target
    binary: str
        A binary variable within the train dataframe
    
    '''
    print(binary, "\n_____________________\n")
    
    ct = pd.crosstab(train[binary], train[categorical_target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, binary, categorical_target)
    mannwhitney = compare_means(train, continuous_target, binary, alt_hyp='two-sided')
    p = plot_cat_by_target(train, categorical_target, binary)
    
    print("\nMann Whitney Test Comparing Means: ", mannwhitney)
    print(chi2_summary)
#     print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
    

    
def run_chi2(train, binary, categorical_target):
    observed = pd.crosstab(train[binary], train[categorical_target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected


def plot_cat_by_target(train, categorical_target, binary):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(categorical_target, binary, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[binary].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

    
def compare_means(train, continuous_target, binary, alt_hyp='two-sided'):
    x = train[train[binary]==0][continuous_target]
    y = train[train[binary]==1][continuous_target]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

###################### ________________________________________
## Bivariate Quant

def explore_bivariate_quant(train, categorical_target, continuous_target, quant):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant, "\n____________________\n")
    descriptive_stats = train.groupby(categorical_target)[quant].describe().T
    spearmans = compare_relationship(train, continuous_target, quant)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, categorical_target, quant)
#     swarm = plot_swarm(train, categorical_target, quant)
    plt.show()
    scatter = plot_scatter(train, categorical_target, continuous_target, quant)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nSpearman's Correlation Test:\n", spearmans)
    print("\n____________________\n")


def compare_relationship(train, continuous_target, quant):
    return stats.spearmanr(train[quant], train[continuous_target], axis=0)

def plot_swarm(train, categorical_target, quant):
    average = train[quant].mean()
    p = sns.swarmplot(data=train, x=categorical_target, y=quant, color='lightgray')
    p = plt.title(quant)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, categorical_target, quant):
    average = train[quant].mean()
    p = sns.boxenplot(data=train, x=categorical_target, y=quant, color='lightseagreen')
    p = plt.title(quant)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_scatter(train, categorical_target, continuous_target, quant):
    p = sns.scatterplot(x=quant, y=continuous_target, hue=categorical_target, data=train)
    p = plt.title(quant)
    return p


######################### ____________________________________

### Multivariate 

#***** Under Construction


def explore_multivariate(train, categorical_target, binary_vars, quant_vars):
    '''
    '''
#     plot_swarm_grid_with_color(train, categorical_target, binary_vars, quant_vars)
    violin = plot_violin_grid_with_color(train, categorical_target, binary_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=categorical_target)
    plt.show()
    plot_all_continuous_vars(train, categorical_target, quant_vars)
    plt.show()    


def plot_all_continuous_vars(train, categorical_target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing survived. 
    '''
    my_vars = [item for sublist in [quant_vars, [categorical_target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=categorical_target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=categorical_target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()
    
def plot_violin_grid_with_color(train, categorical_target, binary_vars, quant_vars):
    for quant in quant_vars:
        sns.violinplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_vars, palette="Set2")
        plt.show()
        
def plot_swarm_grid_with_color(train, categorical_target, binary_vars, quant_vars):
    for quant in quant_vars:
        sns.swarmplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_vars, palette="Set2")
        plt.show()


        