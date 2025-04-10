#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy.io
import os
from datetime import datetime, timedelta
from PIL import Image

# ===== UTILITY FUNCTIONS =====

def convert_date(Matlab_date):
    """
    Convert Matlab serial date number to Python datetime.
    
    Parameters:
    -----------
    Matlab_date : float
        Matlab serial date number
        
    Returns:
    --------
    datetime or None
        Python datetime object or None if conversion fails
    """
    try:
        python_dt = datetime.fromordinal(int(Matlab_date)) + timedelta(days=Matlab_date%1) - timedelta(days = 366)
    except OverflowError:
        python_dt = None
    return python_dt

def select_first_element(array):
    """
    Select the first element from a numpy array, or return NaN if empty.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array to extract first element from
        
    Returns:
    --------
    object or numpy.nan
        First element of the array or NaN if array is empty
    """
    if len(array) > 0:
        return array[0]
    else:
        return np.nan

def calculate_age(df):
    """
    Calculate age of each person in the dataframe based on birth date and photo year.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'year_photo_taken' and 'dob_py' columns
        
    Returns:
    --------
    list
        List of calculated ages for each row in the DataFrame
    """
    age_list = []
    counter = 0
    for index, row in df.iterrows():
        counter += 1
        # Since data of photo is not known precisely, it is assumed to be July 1st
        date_photo = datetime(year = row["year_photo_taken"], month = 7, day = 1)
        date_of_birth = row["dob_py"]
        if date_of_birth.month < date_photo.month:
            age = int(date_photo.year - date_of_birth.year)
        else:
            age = int(date_photo.year - date_of_birth.year - 1)
        age_list.append(age)
    return age_list

def clarify_gender(gender):
    """
    Convert numeric gender values to string labels.
    
    Parameters:
    -----------
    gender : float
        Gender value (1.0 for male, 0.0 for female)
        
    Returns:
    --------
    str
        String representation of gender ('male' or 'female')
    """
    if gender == 1.0:
        gender = "male"
    else:
        gender = "female"
    return gender

def define_new_file_name(row):
    """
    Create a descriptive file name based on dataframe row.
    
    Parameters:
    -----------
    row : pandas.Series
        Row from DataFrame containing name, gender, and age information
        
    Returns:
    --------
    str
        New file name in the format "{index}_{name}_{gender}_{age}.jpg"
    """
    return str(row.name)+"_"+row["name"]+"_"+row["gender"]+"_"+str(row["age"])+".jpg"

def define_new_path(row, path, feature):
    """
    Define path for storing processed images based on specified feature.
    
    Parameters:
    -----------
    row : pandas.Series
        Row from DataFrame containing relevant information
    path : str
        Base path for the new file
    feature : str
        Column name in row to use for subfolder
        
    Returns:
    --------
    str
        Full path for the new file
    """
    return path+row[feature]+"/"+str(row.name)+"_"+row["name"]+"_"+row["gender"]+"_"+str(row["age"])+".jpg"

def stratify_age_data(df, ranges_age = [10, 20, 30, 40, 50, 60, 70, 100], sample_size = 1000):
    """
    Stratify data by age ranges to reduce bias in age distribution.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing an 'age' column
    ranges_age : list
        List of age boundaries for stratification
    sample_size : int
        Maximum number of samples to take from each age range
        
    Returns:
    --------
    pandas.DataFrame
        Stratified DataFrame
    """
    new_df = pd.DataFrame(columns = df.columns)
    df = df[(df["age"] >= ranges_age[0]) & (df["age"] < ranges_age[-1])]
    for index, age in enumerate(ranges_age[:-1]):
        df_age = df[(df["age"] > ranges_age[index]) & (df["age"] < ranges_age[index +1])]
        if len(df_age) > sample_size:
            df_age = df_age.sample(n = sample_size)
        new_df = pd.concat([new_df, df_age])
    return new_df

def is_verified(row, folder, feature):
    """
    Check if an image exists at the specified path.
    
    Parameters:
    -----------
    row : pandas.Series
        Row from DataFrame containing 'new_path'
    folder : str
        Base folder containing verified images
    feature : str
        Feature used for subfolder name
        
    Returns:
    --------
    int
        1 if file exists, 0 otherwise
    """
    if os.path.exists(row["new_path"]):
        return 1
    else:
        return 0

def count_pixels(df):
    """
    Count the number of pixels in each image.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'new_path' column with image paths
        
    Returns:
    --------
    list
        List of pixel counts for each image
    """
    nb_pixels_image_list = []
    for index, row in df.iterrows():
        with Image.open(row["new_path"]) as img:
            width, height = img.size
            nb_pixels_image_list.append(width * height)
    return nb_pixels_image_list

# ===== DATA PROCESSING FUNCTIONS =====

def convert_mat_dataframe(mat_file, columns, dict_key):
    """
    Convert a MATLAB .mat file to a pandas DataFrame.
    
    Parameters:
    -----------
    mat_file : str
        Path to the .mat file
    columns : list
        List of column names for the DataFrame
    dict_key : str
        Key in the .mat file structure to extract data from
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the data from the .mat file
    """
    mat = scipy.io.loadmat(mat_file)
    mat_data = mat[dict_key][0][0]
    df = pd.DataFrame()
    for index, item in enumerate(mat_data):
        if len(item[0]) == len(mat_data[0][0]): # Columns are only added to the dataframe if all the data is available
            df[columns[index]] = item[0]
    return df

def process_wiki_metadata(mat_file_path, min_age=0, max_age=100, verified_image_path="./data/verified_image/",
                         use_cached=False, cache_path="./data/processed_data.csv"):
    """
    Process the Wiki dataset metadata from a .mat file or load from a cached CSV file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the wiki.mat file
    min_age : int
        Minimum age to include in the processed data
    max_age : int
        Maximum age to include in the processed data
    verified_image_path : str
        Path to the folder containing verified images
    use_cached : bool
        Whether to load data from cache file if it exists
    cache_path : str
        Path to save/load processed data cache
        
    Returns:
    --------
    pandas.DataFrame
        Processed metadata DataFrame
    """
    # Check if cache exists and should be used
    if use_cached and os.path.exists(cache_path):
        # Load the cached data
        wiki_metadata = pd.read_csv(cache_path)
        
        # Convert date column back to datetime
        if 'dob_py' in wiki_metadata.columns:
            wiki_metadata['dob_py'] = pd.to_datetime(wiki_metadata['dob_py'])
        
        # Convert face_location back to numpy arrays if it's stored as string
        if 'face_location' in wiki_metadata.columns and wiki_metadata['face_location'].dtype == 'object':
            def convert_face_location(x):
                if isinstance(x, str):
                    try:
                        # Remove brackets and split by spaces
                        return np.array([float(val) for val in x.strip('[]').split()])
                    except:
                        # If parsing fails, return an empty array
                        return np.array([])
                return x
            
            wiki_metadata['face_location'] = wiki_metadata['face_location'].apply(convert_face_location)
        
        return wiki_metadata
    
    # Definition of columns for metadata files
    columns_wiki = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]
    
    # Converting metadata Matlab file to pandas dataframe
    wiki_metadata = convert_mat_dataframe(mat_file_path, columns_wiki, "wiki")
    
    # Converting date of birth from Matlab serial date number to pandas datetime
    wiki_metadata["dob_py"] = wiki_metadata["dob"].apply(convert_date)
    
    # Removing rows with invalid date of birth
    old_size_wiki = len(wiki_metadata)
    wiki_metadata = wiki_metadata[wiki_metadata["dob_py"].notna()]
    new_size_wiki = len(wiki_metadata)
    
    # Removing missing values and unnecessary numpy array levels
    wiki_metadata["name"] = wiki_metadata["name"].apply(select_first_element)
    wiki_metadata["full_path"] = wiki_metadata["full_path"].apply(select_first_element)
    wiki_metadata["face_location"] = wiki_metadata["face_location"].apply(select_first_element)
    pre_na_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata.dropna(subset=wiki_metadata.drop(["second_face_score"], axis=1).columns)
    post_na_size = len(wiki_metadata)
    
    # Calculating age when photo was taken
    wiki_metadata["age"] = calculate_age(wiki_metadata)
    pre_age_filter_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata[(wiki_metadata["age"] >= min_age) & (wiki_metadata["age"] < max_age)]
    post_age_filter_size = len(wiki_metadata)
    
    # Clarifying gender class
    wiki_metadata["gender"] = wiki_metadata["gender"].apply(clarify_gender)
    
    # Adding new file name and paths
    wiki_metadata["new_file_name"] = wiki_metadata.apply(define_new_file_name, axis=1)
    wiki_metadata["new_path"] = wiki_metadata.apply(define_new_path, args=(verified_image_path, "gender"), axis=1)
    
    # Checking if photo is in verified folder
    wiki_metadata["image_is_verified"] = wiki_metadata.apply(is_verified, args=(verified_image_path, "gender"), axis=1)
    pre_verify_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata[wiki_metadata["image_is_verified"] == 1]
    post_verify_size = len(wiki_metadata)
    
    # Counting number of pixels in each image
    wiki_metadata.loc[:, "number_pixels"] = count_pixels(wiki_metadata)
    
    # Save processed data to cache
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to CSV
    wiki_metadata.to_csv(cache_path, index=False)
    
    return wiki_metadata