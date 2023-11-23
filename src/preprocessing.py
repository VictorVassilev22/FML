import pandas as pd

def set_date_index(df):
    # Check if the DataFrame has a 'Date' column or 'Date' as index
    if 'Date' not in df.columns and df.index.name != 'Date':
        raise ValueError("DataFrame must contain a 'Date' column or index")

    if df.index.name != 'Date':
        df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to pandas datetime objects
        df.set_index('Date', inplace=True)  # Set the 'Date' column as the index
    else:
        df['Date'] = pd.to_datetime(df.index)  # Convert the 'Date' index column to pandas datetime objects

    return df.columns

def remove_features(df, to_remove_list):
    # Get feature column names excluding the target column and columns in to_remove_list
    feature_columns = [col for col in df.columns if col not in to_remove_list]

    # Return feature column names and the target column name
    return feature_columns

def format_col_names(df):
    df.columns = df.columns.str.replace(' ', '_')

def split_features_target(df, target_column, to_remove_list):
    """
    Function to split features and target columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        to_remove_list (list): The list of column names to exclude from the features.

    Returns:
        list: A list of feature column names.
        str: The name of the target column.
    """
    # Get feature column names excluding the target column and columns in to_remove_list
    feature_columns = [col for col in df.columns if col != target_column and col not in to_remove_list]

    # Return feature column names and the target column name
    return feature_columns, target_column

def train_test_split(c, X, y):
    # c is between 0 and 1
    train_size = int(c * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test