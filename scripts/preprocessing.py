import numpy as np
import pandas as pd
import scipy.io
import os
import shutil
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

def sanitize_name(name):
    """
    Replace spaces with hyphens and remove any special characters from name.
    
    Parameters:
    -----------
    name : str
        Name that may contain spaces or special characters
        
    Returns:
    --------
    str
        Sanitized name with spaces replaced by hyphens
    """
    if isinstance(name, str):
        # Replace spaces with hyphens
        sanitized = name.replace(" ", "-")
        # Remove any other problematic characters for filenames
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '-_.')
        return sanitized
    return str(name)

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
    return f"{row.name}_{row['name']}_{row['gender']}_{row['age']}.jpg"

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
    return f"{path}{row[feature]}/{row.name}_{row['name']}_{row['gender']}_{row['age']}.jpg"

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

def copy_verified_images(df, source_base_path, target_base_path="./data/processed/"):
    """
    Copy verified images to a new location with sanitized filenames.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'file_path' and 'sanitized_path' columns
    source_base_path : str
        Base path of the source images
    target_base_path : str
        Base path to copy the images to
    
    Returns:
    --------
    int
        Number of successfully copied images
    """
    # Create target directories if they don't exist
    os.makedirs(os.path.join(target_base_path, "male"), exist_ok=True)
    os.makedirs(os.path.join(target_base_path, "female"), exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    print(f"Copying verified images to {target_base_path}...")
    
    for idx, row in df.iterrows():
        source_path = row["new_path"]  # Original path with original filename
        # Create target path with sanitized filename
        target_path = os.path.join(
            target_base_path, 
            row["gender"], 
            f"{sanitize_name(row['original_name'])}_{row['gender']}_{row['age']}.jpg"
        )
        
        try:
            # Check if the source exists
            if os.path.exists(source_path):
                # Copy the file
                shutil.copy2(source_path, target_path)
                success_count += 1
                
                # Print progress periodically
                if success_count % 100 == 0:
                    print(f"Copied {success_count} images...")
            else:
                error_count += 1
                if error_count < 10:  # Limit error messages to avoid spam
                    print(f"Warning: Source file not found: {source_path}")
                elif error_count == 10:
                    print("Too many errors, suppressing further messages...")
        except Exception as e:
            error_count += 1
            if error_count < 10:  # Limit error messages to avoid spam
                print(f"Error copying {source_path} to {target_path}: {str(e)}")
            elif error_count == 10:
                print("Too many errors, suppressing further messages...")
    
    print(f"Image copying complete. Successfully copied {success_count} images. Encountered {error_count} errors.")
    return success_count

def process_wiki_metadata(mat_file_path, min_age=0, max_age=100, verified_image_path="./data/verified_image/",
                         processed_image_path="./data/processed/", use_cached=False, cache_path="./data/processed_data.csv",
                         copy_images=True):
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
    processed_image_path : str
        Path to copy processed images with sanitized filenames
    use_cached : bool
        Whether to load data from cache file if it exists
    cache_path : str
        Path to save/load processed data cache
    copy_images : bool
        Whether to copy verified images to processed_image_path with sanitized names
        
    Returns:
    --------
    pandas.DataFrame
        Processed metadata DataFrame
    """
    # Check if cache exists and should be used
    if use_cached and os.path.exists(cache_path):
        # Load the cached data
        wiki_metadata = pd.read_csv(cache_path)
        print(f"Loaded cached metadata from {cache_path}")
        print(f"Dataset contains {len(wiki_metadata)} records.")
        return wiki_metadata
    
    print("Processing Wiki metadata from .mat file...")
    
    # Definition of columns for metadata files
    columns_wiki = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]
    
    # Converting metadata Matlab file to pandas dataframe
    wiki_metadata = convert_mat_dataframe(mat_file_path, columns_wiki, "wiki")
    print(f"Initial records from .mat file: {len(wiki_metadata)}")
    
    # Converting date of birth from Matlab serial date number to pandas datetime
    print("Converting dates...")
    wiki_metadata["dob_py"] = wiki_metadata["dob"].apply(convert_date)
    
    # Removing rows with invalid date of birth
    old_size_wiki = len(wiki_metadata)
    wiki_metadata = wiki_metadata[wiki_metadata["dob_py"].notna()]
    new_size_wiki = len(wiki_metadata)
    print(f"Removed {old_size_wiki - new_size_wiki} records with invalid dates. Remaining: {new_size_wiki}")
    
    # Removing missing values and unnecessary numpy array levels
    print("Processing array fields...")
    wiki_metadata["name"] = wiki_metadata["name"].apply(select_first_element)
    wiki_metadata["full_path"] = wiki_metadata["full_path"].apply(select_first_element)
    wiki_metadata["face_location"] = wiki_metadata["face_location"].apply(select_first_element)
    pre_na_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata.dropna(subset=wiki_metadata.drop(["second_face_score"], axis=1).columns)
    post_na_size = len(wiki_metadata)
    print(f"Removed {pre_na_size - post_na_size} records with missing values. Remaining: {post_na_size}")
    
    # Calculating age when photo was taken
    print("Calculating ages...")
    wiki_metadata["age"] = calculate_age(wiki_metadata)
    pre_age_filter_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata[(wiki_metadata["age"] >= min_age) & (wiki_metadata["age"] < max_age)]
    post_age_filter_size = len(wiki_metadata)
    print(f"Removed {pre_age_filter_size - post_age_filter_size} records outside age range [{min_age}, {max_age}). Remaining: {post_age_filter_size}")
    
    # Clarifying gender class
    print("Processing gender information...")
    wiki_metadata["gender"] = wiki_metadata["gender"].apply(clarify_gender)
    
    # Adding new file name and paths with original names (for verification)
    print("Creating file paths for verification...")
    wiki_metadata["new_file_name"] = wiki_metadata.apply(define_new_file_name, axis=1)
    wiki_metadata["new_path"] = wiki_metadata.apply(define_new_path, args=(verified_image_path, "gender"), axis=1)
    
    # Checking if photo is in verified folder
    print("Checking for verified images...")
    wiki_metadata["image_is_verified"] = wiki_metadata.apply(is_verified, args=(verified_image_path, "gender"), axis=1)
    pre_verify_size = len(wiki_metadata)
    wiki_metadata = wiki_metadata[wiki_metadata["image_is_verified"] == 1]
    post_verify_size = len(wiki_metadata)
    print(f"Kept {post_verify_size} verified images out of {pre_verify_size} total.")
    
    # Now sanitize the names (AFTER verification)
    print("Sanitizing filenames (replacing spaces with hyphens)...")
    # Store original name before sanitizing (might be needed for later reference)
    wiki_metadata["original_name"] = wiki_metadata["name"]
    # Sanitize the name column
    wiki_metadata["name"] = wiki_metadata["name"].apply(sanitize_name)
    # Update file names and paths with sanitized names - remove row index from filename
    wiki_metadata["sanitized_file_name"] = wiki_metadata.apply(
        lambda row: f"{sanitize_name(row['original_name'])}_{row['gender']}_{row['age']}.jpg", 
        axis=1
    )
    wiki_metadata["sanitized_path"] = wiki_metadata.apply(
        lambda row: f"{processed_image_path}{row['gender']}/{sanitize_name(row['original_name'])}_{row['gender']}_{row['age']}.jpg", 
        axis=1
    )
    
    # Copy verified images to processed directory with sanitized names if requested
    if copy_images:
        copy_verified_images(wiki_metadata, verified_image_path, processed_image_path)
    
    # Rename columns before dropping unnecessary ones
    wiki_metadata = wiki_metadata.rename(columns={
        'sanitized_file_name': 'file_name', 
        'sanitized_path': 'file_path'
    })
    
    # Drop specified columns
    columns_to_drop = ['dob', 'year_photo_taken', 'full_path', 'original_name', 'face_location', 
                      'face_score', 'second_face_score', 'dob_py', 'image_is_verified',
                      'new_file_name', 'new_path']
    wiki_metadata = wiki_metadata.drop(columns=columns_to_drop)
    
    # Save processed data to cache
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to CSV
    print(f"Saving processed metadata to {cache_path}...")
    wiki_metadata.to_csv(cache_path, index=False)
    print(f"Processing complete. Final dataset contains {len(wiki_metadata)} records.")
    
    return wiki_metadata