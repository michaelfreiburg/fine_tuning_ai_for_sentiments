import pandas as pd

class DataCleanerAndRefiner:
    """
    class to clean and refine a dataset with sentiment classifications obtained via the chat GPT API 
    by separating valid from invalid data, 
    refining the valid data by mapping their classification texts to integers as labels
    and saving both the valid, refined results as well as the invalid results.
    
    parameters:
    - valid_values (list): the valid values used to filter the dataset
    - column_names (list): the columns which shall be processed
    - sentiment_label_conversion (dict): a mapping from sentiment text to numerical labels e.g. {"positive":0, "neutral":1, "negative":2}
    """
    
    def __init__(self, valid_values: list, column_names: list, sentiment_label_conversion: dict):
        """
        initializing the DataCleanerAndRefiner
        """
        self.valid_values = valid_values
        self.column_names = column_names
        self.sentiment_label_conversion = sentiment_label_conversion

    def _separate_valid_from_invalid_data(self, df_gpt_output: pd.DataFrame):
        """
        separates valid from invalid data based on the predefined valid values.
        Valid data contains only valid values across all the specified columns 
        
        arguments:
        - df_gpt_output (pd.DataFrame): the input DataFrame to be processed
        
        returns:
        - tuple: a tuple containing two DataFrames (df_valid, df_invalid)
        """
        # convert all values of the selected columns to lower case to allow case-insensitive comparison
        df_gpt_output[self.column_names] = df_gpt_output[self.column_names].apply(lambda x: x.str.lower())
        
        # check which rows have only valid values in the selected columns   
        valid_gpt_output = df_gpt_output[self.column_names].isin(self.valid_values).all(axis=1)
        
        # creating a DataFrame each for the valid and the invalid data
        df_valid = df_gpt_output[valid_gpt_output]
        df_invalid = df_gpt_output[~valid_gpt_output]

        return df_valid, df_invalid        

    def _refine_valid_data(self, df_valid: pd.DataFrame):
        """
        refines the valid data by converting the classification from text to numerical labels applying a mapping

        arguments:
        - df_valid (pd.DataFrame): the DataFrame containing the valid data which shall be refined
        
        returns:
        - pd.DataFrame: the refined DataFrame with the classifications labels as integers
        """
        # applying the conversion to the specified columns
        return df_valid[self.column_names].map(self.sentiment_label_conversion)

    def _save_refined_valid_and_invalid_data(self, refined_valid_path_and_filename, invalid_path_and_filename, df_valid_refined, df_invalid):
        """
        saves the valid and invalid data into separate csv files
        
        arguments:
        - refined_valid_path_and_filename (str): the path and filename to save the valid refined data.
        - invalid_path_and_filename (str): the path and filename to save the invalid data
        - df_valid_refined (pd.DataFrame): the DataFrame with the valid refined data
        - df_invalid (pd.DataFrame): the DataFrame with the invalid data
        """
        # saving the valid and refined data
        df_valid_refined.to_csv(refined_valid_path_and_filename, index=False)     
        
        # saving the invalid data only if it exists
        if len(df_invalid) > 0:
            df_invalid.to_csv(invalid_path_and_filename, index=False)
        else:
            print(f"No invalid data found and therefore no file for invalid data was created")
            
    def clean_and_refine_data(self, df_raw, refined_valid_path_and_filename, invalid_path_and_filename):    
        """
        orchestrates the full data cleaning and refining process:
        1. separates valid and invalid data
        2. refines the valid data by converting its labels
        3. saves both valid refined and invalid data to separate CSV files 
        
        arguments:
        - df_raw (pd.DataFrame): the raw input DataFrame containing all texts which were returned from the GPT API as sentiments
        - refined_valid_path_and_filename (str): The path and filename to save the valid refined data
        - invalid_path_and_filename (str): The path and filename to save the invalid data.
        """
        # separate valid and invalid data
        df_valid, df_invalid = self._separate_valid_from_invalid_data(df_raw)
        
        # refine the valid data
        df_valid_and_refined = self._refine_valid_data(df_valid)
        
        # save both the valid refined and invalid data (if any exists)
        self._save_refined_valid_and_invalid_data(refined_valid_path_and_filename, invalid_path_and_filename, df_valid_and_refined, df_invalid)