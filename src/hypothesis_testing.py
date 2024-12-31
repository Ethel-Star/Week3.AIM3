# hypothesis_testing.py

import pandas as pd
import scipy.stats as stats

def load_data(file_path):
    """
    Load data from a CSV or other file formats.
    """
    return pd.read_csv(file_path)

def segment_data_by_group(df, column, group_a_value, group_b_value):
    """
    Segment data into two groups based on a column value.
    """
    group_a = df[df[column] == group_a_value]
    group_b = df[df[column] == group_b_value]
    return group_a, group_b

def chi_squared_test(group_a, group_b, column):
    """
    Perform a chi-squared test to compare categorical variables.
    """
    contingency_table = pd.crosstab(group_a[column], group_b[column])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return p_value

def t_test(group_a, group_b, column):
    """
    Perform a t-test to compare the means of two groups for a continuous variable.
    Assumes equal variance by default.
    """
    t_stat, p_value = stats.ttest_ind(group_a[column], group_b[column])
    return p_value

def z_test(group_a, group_b, column):
    """
    Perform a Z-test for two sample means. Assumes large sample size.
    """
    mean_a = group_a[column].mean()
    mean_b = group_b[column].mean()
    std_a = group_a[column].std()
    std_b = group_b[column].std()
    n_a = len(group_a)
    n_b = len(group_b)
    
    z_stat = (mean_a - mean_b) / ( (std_a**2 / n_a + std_b**2 / n_b)**0.5 )
    p_value = stats.norm.sf(abs(z_stat)) * 2  # two-tailed test
    return p_value

def run_hypothesis_testing(df, column, group_a_value, group_b_value, test_type='t-test'):
    """
    Main function to run hypothesis tests.
    - 'test_type' can be 't-test', 'z-test', or 'chi-squared'.
    """
    group_a, group_b = segment_data_by_group(df, column, group_a_value, group_b_value)
    
    if test_type == 'chi-squared':
        return chi_squared_test(group_a, group_b, column)
    elif test_type == 't-test':
        return t_test(group_a, group_b, column)
    elif test_type == 'z-test':
        return z_test(group_a, group_b, column)
    else:
        raise ValueError("Invalid test type. Choose from 'chi-squared', 't-test', or 'z-test'.")

def analyze_p_value(p_value):
    """
    Analyze p-value and return conclusion based on 0.05 threshold.
    """
    if p_value < 0.05:
        return "Reject the null hypothesis. There is a significant difference."
    else:
        return "Fail to reject the null hypothesis. There is no significant difference."
