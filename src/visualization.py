# src/visualization.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def outlier_detection(df: pd.DataFrame):
    """
    Detects and visualizes outliers using box plots.
    
    :param df: DataFrame with numerical columns
    """
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Outlier Detection for {col}')
        plt.show()

def trends_over_geography(df: pd.DataFrame, column: str, geography_column: str):
    """
    Visualizes trends over geography for a given column.
    
    :param df: DataFrame with data
    :param column: The column to analyze
    :param geography_column: The geographical column to group by
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=geography_column, y=column, data=df)
    plt.title(f'{column} Trends by {geography_column}')
    plt.show()

def visualize_eda(df: pd.DataFrame):
    """
    Creates 3 creative and beautiful plots to visualize the key insights from EDA.
    
    :param df: DataFrame with the EDA results
    """
    # Example visualizations (you can customize these based on your analysis)
    plt.figure(figsize=(12, 8))
    sns.histplot(df['TotalPremium'], kde=True)
    plt.title('Distribution of Total Premium')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['TotalPremium'], y=df['TotalClaim'])
    plt.title('Total Premium vs Total Claim')
    plt.show()
