# Import libs
import pandas as pd
import scipy
import numpy as np
import os
import PIL

from datetime import datetime, timedelta
from PIL import ImageEnhance

# This function converts metadata Matlab files to pandas dataframe
def convert_mat_dataframe(mat_file, columns, dict_key):
    mat = scipy.io.loadmat(mat_file)
    mat_data = mat[dict_key][0][0]
    df = pd.DataFrame()
    for index, item in enumerate(mat_data):
        if len(item[0]) == len(mat_data[0][0]): # Columns are only added to the dataframe if all the data is available
            df[columns[index]] = item[0]
    return(df)

# This function converts the Matlab serial date number to pandas datetime
def convert_date(Matlab_date):
    try:
        python_dt = datetime.fromordinal(int(Matlab_date)) + timedelta(days=Matlab_date%1) - timedelta(days = 366)
    except OverflowError:
        python_dt = None
    return python_dt

# This function converts the numpy arrays in the metadata Matlab files to its first element
def select_first_element(array):
    if len(array) > 0:
        return array[0]
    else:
        return np.nan

# This function calculates the age of the person the year the photo was taken
def calculate_age(df):
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

# This function clarifies the gender in the daraframe (1.0 is replaced by "male" and 0.0 is replaced by "female")
def clarify_gender(gender):
    if gender == 1.0:
        gender = "male"
    else:
        gender = "female"
    return gender

# This function defines a more explicit file name for the image file
def define_new_file_name(row):
    return str(row.name)+"_"+row["name"]+"_"+row["gender"]+"_"+str(row["age"])+".jpg"

# This function defines the new training folder for the image depending on the selected feature
def define_new_path(row, path, feature):
    return path[6:]+row[feature]+"/"+str(row.name)+"_"+row["name"]+"_"+row["gender"]+"_"+str(row["age"])+".jpg"

# This function filters the dataframe to ensure that the range_age is evenly distributed (to avoid bias)
def stratify_age_data(df, ranges_age = [10, 20, 30, 40, 50, 60, 70, 100], sample_size = 1000):
    new_df = pd.DataFrame(columns = df.columns)
    df = df[(df["age"] >= ranges_age[0]) & (df["age"] < ranges_age[-1])]
    for index, age in enumerate(ranges_age[:-1]):
        df_age = df[(df["age"] > ranges_age[index]) & (df["age"] < ranges_age[index +1])]
        if len(df_age) > sample_size:
            df_age = df_age.sample(n = sample_size)
        new_df = new_df.append(df_age)
    return new_df

# This function verifies if the picture is in the verified_data folder
def is_verified(row, folder, feature):
    if os.path.exists("../../" + row["new_path"]):
        return 1
    else:
        return 0

# This function calculates the number of pixels in each image
def count_pixels(df):
    nb_pixels_image_list = []
    for index, row in df.iterrows():
        with PIL.Image.open("../../" + row["new_path"]) as img:
            width, height = img.size
            nb_pixels_image_list.append(width * height)
    return nb_pixels_image_list

# Definition of columns for metadata files
columns_wiki = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

# Minimum and maximum age for age recognition (to remove absurd values such as centuries old portraits)
min_age = 0
max_age = 100

# Converting metadata Matlab file to pandas dataframe
wiki_metadata = convert_mat_dataframe("../../data/wiki/wiki.mat", columns_wiki, "wiki")

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
wiki_metadata = wiki_metadata.dropna(subset = wiki_metadata.drop(["second_face_score"], axis = 1).columns)

# Calculating age when photo was taken
wiki_metadata["age"] = calculate_age(wiki_metadata)
wiki_metadata = wiki_metadata[(wiki_metadata["age"] >= min_age) & (wiki_metadata["age"] < max_age)]

# Clarifying gender class
wiki_metadata["gender"] = wiki_metadata["gender"].apply(clarify_gender)

# Adding new file name and paths
wiki_metadata["new_file_name"] = wiki_metadata.apply(define_new_file_name, axis = 1)
wiki_metadata["new_path"] = wiki_metadata.apply(define_new_path , args = ("../../data/verified_image/", "gender", ), axis = 1)

# Checking if photo is in verified folder
wiki_metadata["image_is_verified"] = wiki_metadata.apply(is_verified, args = ("../../data/verified_image/", "gender", ), axis = 1)
wiki_metadata = wiki_metadata[wiki_metadata["image_is_verified"] == 1]

# Counting number of pixels in each image
wiki_metadata.loc[:, "number_pixels"] = count_pixels(wiki_metadata)