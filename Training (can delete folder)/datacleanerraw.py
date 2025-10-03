import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def clean_and_split_data(file_path, skiprows, target_col):
    """
    Loads, cleans, and splits a dataset for machine learning, without resampling.
    """
    try:
        df = pd.read_csv(file_path, skiprows=skiprows)
        print(f"--- Successfully loaded {file_path} ---")

        
        df.dropna(axis=1, how='all', inplace=True)

       
        object_cols = df.select_dtypes(include=['object']).columns
        cols_to_drop = [col for col in object_cols if col != target_col]
        df.drop(columns=cols_to_drop, inplace=True)

        
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data preparation complete.")
        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None



if __name__ == "__main__":
   
    output_dir = 'processed_data_clean'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   
    print("\n--- Preparing Kepler Data (without SMOTE) ---")
    kepler_X_train, kepler_X_test, kepler_y_train, kepler_y_test = clean_and_split_data(
        'koi.csv', 53, 'koi_disposition'
    )

    
    if kepler_X_train is not None:
        kepler_X_train.to_csv(os.path.join(output_dir, 'kepler_X_train.csv'), index=False)
        kepler_X_test.to_csv(os.path.join(output_dir, 'kepler_X_test.csv'), index=False)
        kepler_y_train.to_csv(os.path.join(output_dir, 'kepler_y_train.csv'), index=False)
        kepler_y_test.to_csv(os.path.join(output_dir, 'kepler_y_test.csv'), index=False)
        print("Kepler data saved successfully.")

   
    print("\n--- Preparing TESS Data (without SMOTE) ---")
    tess_X_train, tess_X_test, tess_y_train, tess_y_test = clean_and_split_data(
        'toi.csv', 69, 'tfopwg_disp'
    )

   
    if tess_X_train is not None:
        tess_X_train.to_csv(os.path.join(output_dir, 'tess_X_train.csv'), index=False)
        tess_X_test.to_csv(os.path.join(output_dir, 'tess_X_test.csv'), index=False)
        tess_y_train.to_csv(os.path.join(output_dir, 'tess_y_train.csv'), index=False)
        tess_y_test.to_csv(os.path.join(output_dir, 'tess_y_test.csv'), index=False)
        print("TESS data saved successfully.")