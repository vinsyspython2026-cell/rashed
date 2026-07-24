import pickle
import pandas as pd

def extract(data_source):
    """Extracts data from a CSV file (local or URL) or converts a single row of data into a pandas DataFrame.

    Args:
        data_source: Can be a file path (str), a URL (str), or a dictionary/list representing a single row.

    Returns:
        pd.DataFrame: The extracted or created DataFrame.
    """
    try:
        if isinstance(data_source, str):
            if data_source.startswith(('http://', 'https://')):
                # Assume it's a URL
                new_data_df = pd.read_csv(data_source)
                print(f"Successfully extracted data from URL: {data_source}.")
            else:
                # Assume it's a local file path
                new_data_df = pd.read_csv(data_source)
                print(f"Successfully extracted data from file: {data_source}.")
        elif isinstance(data_source, (dict, list)):
            # Assume it's a single row of data, convert to DataFrame
            if isinstance(data_source, dict):
                new_data_df = pd.DataFrame([data_source])
            else: # Assuming list of dictionaries for multiple rows or list of values for a single row
                # If it's a list of lists/values, you might need column names, for simplicity, assuming a list of dicts or single dict
                new_data_df = pd.DataFrame(data_source)
            print(f"Successfully created DataFrame from single row data.")
        else:
            print(f"Error: Unsupported data source type. Expected string (path/URL) or dict/list. Got {type(data_source)}.")
            return None

        return new_data_df
    except FileNotFoundError:
        print(f"Error: File not found at {data_source}")
        return None
    except Exception as e:
        print(f"An error occurred during data extraction: {e}")
        return None

def transform(new_data_df):
    """Applies data cleaning and transformation steps to new data.

    Args:
        new_data_df (pd.DataFrame): The raw DataFrame of new data.

    Returns:
        pd.DataFrame: The transformed DataFrame ready for prediction.
    """
    # Define the parameters needed for the transform function, derived from the training data
    # These values are hardcoded from the training data statistics.
    median_credit_score = 708.0
    median_max_open_credit = 419507.0
    median_bankruptcies = 0.0
    median_tax_liens = 0.0
    mode_years_in_job = '10+ years'

    term_mapping = {'Short Term': 0, 'Long Term': 1}

    years_in_job_order = [
        '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
        '6 years', '7 years', '8 years', '9 years', '10+ years'
    ]
    years_in_job_mapping = {val: i for i, val in enumerate(years_in_job_order)}

    # model_features are the selected_features from the training phase.
    # This list should also be loaded or known from the trained model's requirements.
    model_features = ['Tax Liens', 'Term_Encoded', 'Home Ownership_Own Home',
                           'Home Ownership_Rent', 'Purpose_Other', 'Credit Score']

    processed_df = new_data_df.copy()

    # Null Value Treatment
    processed_df['Credit Score'] = processed_df['Credit Score'].fillna(median_credit_score)
    processed_df['Maximum Open Credit'] = processed_df['Maximum Open Credit'].fillna(median_max_open_credit)
    processed_df['Bankruptcies'] = processed_df['Bankruptcies'].fillna(median_bankruptcies)
    processed_df['Tax Liens'] = processed_df['Tax Liens'].fillna(median_tax_liens)
    processed_df['Years in current job'] = processed_df['Years in current job'].fillna(mode_years_in_job)

    print("Null values processed.")

    # Drop Unwanted Columns
    cols_to_drop = ['Loan ID', 'Customer ID']
    processed_df = processed_df.drop(columns=[col for col in cols_to_drop if col in processed_df.columns])
    print("Dropped 'Loan ID' and 'Customer ID'.")

    # Value Correction for 'Home Ownership'
    if 'Home Ownership' in processed_df.columns:
        processed_df['Home Ownership'] = processed_df['Home Ownership'].replace('HaveMortgage', 'Home Mortgage')
        print("Corrected 'Home Ownership' values.")

    # Encoding
    # Ordinal encoding for 'Term'
    if 'Term' in processed_df.columns:
        processed_df['Term_Encoded'] = processed_df['Term'].map(term_mapping)

    # Ordinal encoding for 'Years in current job'
    if 'Years in current job' in processed_df.columns:
        processed_df['Years in current job_Encoded'] = processed_df['Years in current job'].map(years_in_job_mapping)

    # Drop the original 'Term' and 'Years in current job' columns after encoding
    cols_to_drop_after_ordinal = ['Term', 'Years in current job']
    processed_df = processed_df.drop(columns=[col for col in cols_to_drop_after_ordinal if col in processed_df.columns])

    # One-hot encoding for Home Ownership and Purpose
    cols_for_one_hot = ['Home Ownership', 'Purpose']
    # Only apply one-hot if the column exists and hasn't been dropped
    cols_for_one_hot = [col for col in cols_for_one_hot if col in processed_df.columns]

    if cols_for_one_hot:
        processed_df = pd.get_dummies(processed_df, columns=cols_for_one_hot, drop_first=True, dtype=int)
        print("Applied encoding.")

    # Align columns with the features the model was trained on
    # Reindex to ensure all model_features are present, filling missing with 0
    final_df = processed_df.reindex(columns=model_features, fill_value=0)

    print(f"Transformed data shape: {final_df.shape}")
    print("Data transformation complete.")
    return final_df

def load(data_to_load_df, output_file_path):
    """Loads the processed DataFrame into a CSV file.

    Args:
        data_to_load_df (pd.DataFrame): The DataFrame to save.
        output_file_path (str): The path where the CSV file should be saved.
    """
    try:
        data_to_load_df.to_csv(output_file_path, index=False)
        print(f"Successfully loaded data to {output_file_path}")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")

def predict(transformed_data_df, model_file_path):
    """Loads the saved model and makes predictions on the transformed data.

    Args:
        transformed_data_df (pd.DataFrame): The DataFrame with transformed features.
        model_file_path (str): The path to the saved model pickle file.

    Returns:
        tuple: A tuple containing predictions (numpy array) and prediction probabilities for class 1 (numpy array).
    """
    try:
        # Load the saved model
        with open(model_file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        print(f"Model loaded successfully from {model_file_path}.")

        # Make predictions
        predictions = loaded_model.predict(transformed_data_df)
        # Get prediction probabilities for the positive class (class 1: Charged Off)
        prediction_proba = loaded_model.predict_proba(transformed_data_df)[:, 1]

        print("Predictions generated.")
        return predictions, prediction_proba
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

