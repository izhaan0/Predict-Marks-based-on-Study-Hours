import pandas as pd 

def load_data(file_path):
    """
    Load dataset from the specified file path.

    """
    df=pd.read_csv(file_path)
    return df

def check_nulls(df):
    """    
    Check for missing values in the dataset.

    """
    print("Checking for missing values...")
    print(df.isnull().sum())
    if df.isnull().values.any():
        print("\n⚠️Missing values found. Consider handling them before proceeding.")
        missing_data = df[df.isnull().any(axis=1)]
        print("\nRows with missing values: ")
        print(missing_data)
        
    else:
        print("\n✅No missing values found in the dataset.")
    
def clean_data(df , strategy="drop"):
    """    
    Handling missing values and duplicates in the dataset.
    """
    if df.isnull().values.any():
        if strategy == 'drop':
            df=df.dropna()
            print("Dropped rows with missing values.")
        elif strategy == 'fill_mean':
            df=df.fillna(df.mean())
            print("Filled missing values with column means.")
    
    else:
        print("No Cleaning needed . Dataset is already clean.") 
    
    return df

def get_feature_target(df):
    """    
    Get feature and target variables from the dataset.

    """
    X=df[["Hours"]]    # feature
    y=df["Scores"]      # target 
    
    return X, y

# Example usage
if __name__ == "__main__":
    file_path = "./data/score_updated.csv"
    df = load_data(file_path)
    print("Dataset loaded successfully.")
    check_nulls(df)
    df = clean_data(df, strategy='drop')
    X, y = get_feature_target(df)
    print("Feature and target variables extracted successfully.")
    


    