import pandas as pd

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean'):
    """
    Handles missing values by using the specified strategy.
    """
    if strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        df.fillna(df.mode().iloc[0], inplace=True)
    elif strategy == 'drop':
        df.dropna(inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df

def handle_outliers(df: pd.DataFrame, threshold: float = 1.5):
    """
    Detects and caps outliers based on IQR.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Cap outliers instead of dropping them
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].clip(lower=lower_bound[col], upper=upper_bound[col])
    
    return df
