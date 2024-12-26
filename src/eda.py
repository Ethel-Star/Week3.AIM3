# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_summary(df: pd.DataFrame):
    """
    Generates summary statistics for the DataFrame.
    
    :param df: DataFrame to summarize
    """
    print("Data Summary:\n")
    print(df.describe())
    print("\nData Structure:\n")
    print(df.info())

def univariate_analysis(df: pd.DataFrame):
    """
    Performs univariate analysis using histograms and bar charts.
    
    :param df: DataFrame to analyze
    """
    # Histograms for numerical columns
    df.hist(figsize=(10, 10), bins=30)
    plt.show()

    # Bar charts for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col].value_counts().plot(kind='bar')
        plt.title(f'{col} Distribution')
        plt.show()

def bivariate_analysis(df: pd.DataFrame, feature1: str, feature2: str):
    """
    Explores relationships between two features using scatter plots and correlation matrix.
    
    :param df: DataFrame containing the data
    :param feature1: First feature for the analysis
    :param feature2: Second feature for the analysis
    """
    # Scatter plot
    df.plot(kind='scatter', x=feature1, y=feature2)
    plt.title(f'{feature1} vs {feature2}')
    plt.show()

    # Correlation matrix
    correlation = df[[feature1, feature2]].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'{feature1} and {feature2} Correlation')
    plt.show()
def descriptive_statistics(df: pd.DataFrame):
    """
    Calculates and returns the descriptive statistics for numerical features.

    :param df: DataFrame with the data to analyze
    :return: DataFrame with descriptive statistics for numerical columns
    """
    # Selecting only numerical columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculating descriptive statistics for numerical columns
    numerical_stats = numeric_df.describe().T
    
    # Calculating additional statistics like variance and standard deviation
    numerical_stats['variance'] = numeric_df.var()
    numerical_stats['std_dev'] = numeric_df.std()
    
    return numerical_stats
# src/eda.py
def check_data_structure(df: pd.DataFrame):
    """
    Checks the data types of columns in the DataFrame and ensures that categorical variables, dates, etc., 
    are properly formatted.

    :param df: DataFrame with the data to check
    :return: DataFrame with column names and their data types
    """
    # Getting the data types of each column
    column_dtypes = df.dtypes
    
    # Checking for categorical variables (usually object type)
    categorical_columns = df.select_dtypes(include=['object']).columns
    datetime_columns = df.select_dtypes(include=['datetime']).columns

    # Report structure information
    data_structure_report = {
        'column_dtypes': column_dtypes,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns
    }

    return data_structure_report
