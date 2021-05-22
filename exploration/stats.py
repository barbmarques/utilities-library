import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
import datetime as dt



##########################################################################################

# Stats

##########################################################################################

def run_stats_on_everything(train, categorical_target, continuous_target, binary_vars, quant_vars):
    '''
    
    Description:
    -----------
    This function takes in the train dataframe and the segregated columns and runs statistical tests based on the variable type.
    
    Parameters:
    ----------
    train: df
        train dataframe
    categorical_target: str
        String of the categorical target variable
    continuous_target: str
        String of the continuous target variable
    binary_vars: str or list of str
        String or list of variable that are binary
    quant_vars: str or list
        String or list of variables that are continuous
    
    '''
    
    # Cycles through binary variables creates a crosstab, runs a chi2 test and a manwhitney
    for binary in binary_vars:
        
        ct = pd.crosstab(train[binary], train[categorical_target], margins=True)
        chi2_summary, observed, expected = run_chi2(train, binary, categorical_target)
        mannwhitney = compare_means(train, continuous_target, binary, alt_hyp='two-sided')
        
        # prints results 
        print(binary, "\n_____________________\n")
        print("\nMann Whitney Test Comparing Means: ", mannwhitney)
        print(chi2_summary)
    #     print("\nobserved:\n", ct)
        print("\nexpected:\n", expected)
        print("\n_____________________\n")
    
    
    plt.figure(figsize=(16,12))
    sns.heatmap(train.corr(), cmap='BuGn')
    plt.show()
    
    # Cycles through quantitative variables runs spearmans correlation against continuous targets
    for quant in quant_vars:

        spearmans = compare_relationship(train, continuous_target, quant)
        
        # Prints results
        print(quant, "\n____________________\n")
        print("Spearman's Correlation Test:\n")
        print(spearmans)
        print("\n____________________")
        print("____________________\n")

        
        
        
        
def t_test(population_1, population_2, alpha=0.05, sample=1, tail=2, tail_dir='higher'):
    '''
    
    Description:
    -----------
    This function takes in 2 populations, and an alpha confidence level and outputs the results of a t-test.
    
    Parameters:
    ----------
    population_1: Series
        A series that is a subgroup of the total population. 
    population_2: Series
        When sample = 1, population_2 must be a series that is the total population; 
        When sample = 2,  population_2 can be another subgroup of the same population
    alpha: float
        Default = 0.05, 0 < alpha < 1, Alpha value = 1 - confidence level 
    sample: {1 or 2}, 
        Default = 1, functions performs 1 or 2 sample t-test.
    tail: {1 or 2}, 
        Default = 2, Need to be used in conjuction with tail_dir. performs a 1 or 2 sample t-test. 
    tail_dir: {'higher' or 'lower'}, 
        default = 'higher'
        
    '''
    
    # One sample, two tail T-test
    if sample == 1 and tail == 2:
        
        # run stats.ttest_1samp
        t, p = stats.ttest_1samp(population_1, population_2.mean())
        
        # prints t-statistic and p value of test
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        
        # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value
        if p < alpha:
            print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, we can reject the null hypothesis')
        else:
            print('There is insufficient evidence to reject the null hypothesis')
    
    # One sample, one tail T-test
    elif sample==1 and tail == 1:
        
        # run stats.ttest_1samp
        t, p = stats.ttest_1samp(population_1, population_2.mean())
        
        # prints t-statistic and p value of test
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        
        # sets the direction to check the if population_1 is greater than the total population
        if tail_dir == "higher":
            
            # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value and the t-statistic
            if (p/2) < alpha and t > 0:
                print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, and the t-stat: {round(t,4)} is greater than 0, we can reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
        
        # sets the direction to check the if population_1 is lower than the total population
        elif tail_dir == "lower":
            
            # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value and the t-statistic
            if (p/2) < alpha and t < 0:
                print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, and the t-stat: {round(t,4)} is less than 0, we can reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
    
    # Two sample, Two tailed T-test
    elif sample==2 and tail == 2:
        
        # run stats.ttest_ind on two subgroups of the total population
        t, p = stats.ttest_ind(population_1, population_2)
    
        # prints t-statistic and p value of test
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        
        # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value
        if p < alpha:
            print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, we reject the null hypothesis')
        else:
            print('There is insufficient evidence to reject the null hypothesis')
    
    # Two sample, One tailed T-test
    elif sample == 2 and tail == 1:
        
        # run stats.ttest_ind on two subgroups of the total population
        t, p = stats.ttest_ind(population_1, population_2)
        
        # prints t-statistic and p value of test
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        
        # sets the direction to check the if population_1 is greater than population_2
        if tail_dir == "higher":
            
            # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value and the t-statistic
            if (p/2) < alpha and t > 0:
                print(f'Because the p-value: {round(p, 4)} is less than alpha: {alpha}, and t-stat: {round(t,4)} is greater than 0, we reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
        
        # sets the direction to check the if population_1 is lower than population_2
        elif tail_dir == "lower":
            
            # runs check if the test rejects the null hypothesis or failed to reject the null hypothesis based on the alpha value and the t-statistic
            if (p/2) < alpha and t < 0:
                print(f'Because the p-value: {round(p, 4)} is less than alpha: {alpha} and the t-stat: {round(t,4)} is less than 0, we reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
    
    # Prints instructions to fix parameters
    else:
        print('sample must be 1 or 2, tail must be 1 or 2, tail_dir must be "higher" or "lower"')
    



def chi2(df, var, target, alpha=0.05):
    '''
    Description:
    -----------
    This function takes in a df, variable, a target variable, and the alpha, and runs a chi squared test. Statistical analysis is printed in the output.
    
    Parameters;
    ---------
    df: Dataframe
    var: str
       Categorical variable to be compared to the target variable
    target: str
        Target categorical variable
    alpha: float
        Default = 0.05, 0 < alpha < 1, Alpha value = 1 - confidence level
        
    '''
    # creates a crosstab of the data
    observed = pd.crosstab(df[var], df[target])
    
    # runs a chi_squared test and returns chi_squared stat, p-value, degrees of freedom, and explected values.
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # Prints the data above
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}\n')
    
    # Tests whether the chi_squared test rejects the null hypothesis or not. 
    if p < alpha:
        print(f'Becasue the p-value: {round(p, 4)} is less than alpha: {alpha}, we can reject the null hypothesis')
    else:
        print('There is insufficient evidence to reject the null hypothesis')
    

