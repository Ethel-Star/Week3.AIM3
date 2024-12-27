# src/visualization.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def outlier_detection(df: pd.DataFrame):
    """
    Detects and visualizes outliers using box plots in a single figure with subplots.
    
    :param df: DataFrame with numerical columns
    """
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a grid of subplots (rows, columns)
    num_plots = len(numeric_cols)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Automatically determine the number of rows
    cols = 3  # Set the number of columns to 3 for better arrangement
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier
    
    # Loop through numerical columns and plot each box plot in the respective subplot
    for i, col in enumerate(numeric_cols):
        # Drop null values before plotting
        cleaned_data = df[col].dropna()
        sns.boxplot(y=cleaned_data, ax=axes[i])  # Use 'y' for single-column boxplots
        axes[i].set_title(f'Outlier Detection for {col}')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def compare_trends_over_geography(df: pd.DataFrame, geography_column: str):
    """
    Compares trends of various features (insurance cover type, premium, auto make) 
    over different geographical regions.
    
    :param df: DataFrame with data
    :param geography_column: The geographical column to group by (e.g., 'Country', 'Province')
    """
    # Plot 1: Comparison of Insurance Cover Type over Geography
    plt.figure(figsize=(12, 8))
    sns.countplot(x=geography_column, hue='CoverType', data=df)
    plt.title(f'Comparison of Insurance Cover Type Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 2: Comparison of Total Premium over Geography
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=geography_column, y='TotalPremium', data=df)
    plt.title(f'Comparison of Total Premium Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 3: Comparison of Auto Make over Geography
    plt.figure(figsize=(12, 8))
    sns.countplot(x=geography_column, hue='make', data=df, order=df['make'].value_counts().index)
    plt.title(f'Comparison of Auto Make Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Optional: Trend of Total Premium over Time (if applicable)
    if 'TransactionMonth' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.lineplot(x='TransactionMonth', y='TotalPremium', hue=geography_column, data=df)
        plt.title(f'Trend of Total Premium Over Time by {geography_column}')
        plt.xticks(rotation=45)
        plt.show()

def visualize_eda(df: pd.DataFrame):
    """
    Creates 3 creative and beautiful plots to visualize the key insights from EDA.
    
    :param df: DataFrame with the EDA results
    """
    # Example visualizations (you can customize these based on your analysis)
    
    # 1. Distribution of Total Premium
    plt.figure(figsize=(12, 8))
    sns.histplot(df['TotalPremium'], kde=True)
    plt.title('Distribution of Total Premium')
    plt.show()

    # 2. Correlation Matrix: Only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # 3. Total Premium vs Total Claim
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['TotalPremium'], y=df['TotalClaims'])
    plt.title('Total Premium vs Total Claims')
    plt.show()
