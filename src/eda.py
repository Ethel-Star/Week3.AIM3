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
    Performs univariate analysis using histograms for numerical columns and bar charts for categorical columns.
    Displays them in a single figure with multiple subplots for better comparison.
    
    :param df: DataFrame to analyze
    """
    # Identifying numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Determine the total number of plots
    num_numerical = len(numerical_columns)
    num_categorical = len(categorical_columns)
    num_plots = num_numerical + num_categorical
    
    # Calculate the number of rows and columns for subplots
    num_cols = 4  # Set 4 columns per row
    num_rows = (num_plots + num_cols - 1) // num_cols  # Round up to the nearest row
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), constrained_layout=True)
    axes = axes.flatten()  # Flatten axes array for easy iteration
    
    # Plot histograms for numerical columns
    for i, col in enumerate(numerical_columns):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Plot bar charts for categorical columns
    for j, col in enumerate(categorical_columns, start=len(numerical_columns)):
        #sns.countplot(data=df, x=col, ax=axes[j], palette='pastel', hue=None, legend=False)
        sns.countplot(data=df, x=col, ax=axes[j], palette='pastel', hue=col, legend=False)

        axes[j].set_title(f'Bar Chart of {col}')
        axes[j].set_xlabel(col)
        axes[j].set_ylabel('Count')
    
    # Hide unused subplots
    for k in range(num_plots, len(axes)):
        axes[k].set_visible(False)
    
    # Show the plot
    plt.show()


def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the DataFrame by renaming columns, converting date columns, 
    and aggregating data.
    """
    # Check and rename 'TransactionMonth' to 'Date'
    if 'TransactionMonth' not in df.columns:
        raise ValueError("DataFrame must contain a 'TransactionMonth' column.")
    
    df.rename(columns={'TransactionMonth': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Aggregating data
    df = df.groupby('Date').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).reset_index()

    # Set 'Date' as index
    df.set_index('Date', inplace=True)
    
    # Resample data by month using Month End frequency
    monthly_data = df.resample('ME').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    })
    
    # Fill missing values using forward fill
    monthly_data.ffill(inplace=True)

    # Check for missing values after forward fill
    missing_values = monthly_data[['TotalPremium', 'TotalClaims']].isnull().sum()
    if missing_values.any():
        print("Missing values found after filling:")
        print(missing_values)

    # Calculate monthly changes
    monthly_data['MonthlyTotalPremiumChange'] = monthly_data['TotalPremium'].pct_change()
    monthly_data['MonthlyTotalClaimsChange'] = monthly_data['TotalClaims'].pct_change()
    
    # Check for missing values in changes
    missing_changes = monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].isnull().sum()
    if missing_changes.any():
        print("Missing values found in MonthlyTotalPremiumChange or MonthlyTotalClaimsChange after calculating changes:")
        print(missing_changes)
    
    # Reset index to bring 'Date' back as a column
    monthly_data.reset_index(inplace=True)
    
    # Debug: Print column names to ensure changes are added
    print("Columns after preprocessing:")
    print(monthly_data.columns)

    return monthly_data

def bivariate_analysis(df, total_premium_column, total_claims_column, postal_code_column):
    """
    Explore relationships between Total Premium and Total Claims as a function of PostalCode 
    using scatter plots and correlation matrices.

    :param df: DataFrame containing the data
    :param total_premium_column: The column name for total premium
    :param total_claims_column: The column name for total claims
    :param postal_code_column: The column name for postal code
    """
    # Ensure required columns exist
    required_columns = [total_premium_column, total_claims_column, postal_code_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Step 1: Group by PostalCode and calculate the mean of TotalPremium and TotalClaims
    postal_code_groups = df.groupby(postal_code_column)[[total_premium_column, total_claims_column]].mean().reset_index()

    # Step 2: Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Scatter plot of TotalPremium vs TotalClaims by PostalCode
    sns.scatterplot(
        x=total_premium_column, 
        y=total_claims_column, 
        hue=postal_code_column, 
        data=postal_code_groups, 
        palette='viridis', 
        s=100, 
        ax=axes[0]
    )
    axes[0].set_title(f'{total_premium_column} vs {total_claims_column} by {postal_code_column}', fontsize=14)
    axes[0].set_xlabel(f'{total_premium_column}', fontsize=12)
    axes[0].set_ylabel(f'{total_claims_column}', fontsize=12)

    # Correlation heatmap between TotalPremium and TotalClaims
    correlation = postal_code_groups[[total_premium_column, total_claims_column]].corr()
    sns.heatmap(
        correlation, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        vmin=-1, 
        vmax=1, 
        center=0, 
        ax=axes[1]
    )
    axes[1].set_title(f'Correlation Matrix: {total_premium_column} & {total_claims_column}', fontsize=14)

    plt.tight_layout()
    plt.show()

    
def compare_data(df):
    """
    Compares trends in insurance cover type, premium, etc., across geographic regions using the 'Province' column.
    """
    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Verify that 'Province' column exists
    if 'Province' not in df.columns:
        raise ValueError("DataFrame must contain a 'Province' column.")
    
    # Verify the data type of 'Province' column
    if df['Province'].dtype != 'object':
        raise ValueError("'Province' column must be of type 'object' (string).")

    # Check if 'TransactionMonth' column exists
    if 'TransactionMonth' not in df.columns:
        raise ValueError("The 'TransactionMonth' column is missing from the DataFrame.")
    
    # Ensure 'TransactionMonth' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Aggregating total premiums by 'Province' and 'TransactionMonth'
    geo_trends = df.groupby(['Province', pd.Grouper(key='TransactionMonth', freq='ME')])['TotalPremium'].sum().unstack()
    # Plotting trends
    plt.figure(figsize=(12, 6))
    for province in geo_trends.columns:
        plt.plot(geo_trends.index, geo_trends[province], label=province)

    plt.title('Trends in Total Premiums by Province')
    plt.xlabel('Month')
    plt.ylabel('Total Premium')
    plt.legend()
    plt.grid(True)
    plt.show()

def detect_outliers(df):
    """
    Uses box plots to detect outliers in numerical data.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns

    plt.figure(figsize=(14, 7))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(3, 4, i)
        sns.boxplot(y=df[column])
        plt.title(f'Boxplot of {column}')

    plt.tight_layout()
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
