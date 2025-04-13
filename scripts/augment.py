import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from typing import Dict, List, Any, Union, Optional, Tuple
import random
import shutil
from deepface import DeepFace

def analyze_age_distribution(
    csv_path: str, 
    max_samples_per_age: int = 500
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze the age distribution in the dataset to determine how many augmentations 
    are needed for each age group and which groups need to be downsampled.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the dataset information
    max_samples_per_age : int
        Maximum number of samples allowed per age
        
    Returns:
    --------
    Dict[int, Dict[str, Any]]
        Dictionary with age as key and augmentation/downsampling information as value, including:
        - 'count': Current count of samples for the age
        - 'augmentation_factor': Number of augmentations to generate per original sample
        - 'samples_to_augment': Number of original samples to augment
        - 'needs_downsampling': Whether this age group needs to be downsampled
        - 'keep_count': Number of samples to keep after downsampling
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Ensure 'age' column exists
    if 'age' not in df.columns:
        raise ValueError("The dataset does not contain an 'age' column")
    
    # Count the number of samples per age
    age_counts = df['age'].value_counts().to_dict()
    
    # Calculate augmentation needs for each age
    age_augmentation_info = {}
    for age, count in age_counts.items():
        if count > max_samples_per_age:
            # Need to downsample
            age_augmentation_info[age] = {
                'count': count,
                'augmentation_factor': 0,
                'samples_to_augment': 0,
                'needs_downsampling': True,
                'keep_count': max_samples_per_age
            }
        elif count == max_samples_per_age:
            # Already at the target
            age_augmentation_info[age] = {
                'count': count,
                'augmentation_factor': 0,
                'samples_to_augment': 0,
                'needs_downsampling': False,
                'keep_count': count
            }
        else:
            # Need to augment
            # Calculate how many more samples we need
            deficit = max_samples_per_age - count
            
            # Apply the 1:3 constraint (original:augmented)
            # This means each original can be augmented to at most 3 new samples
            max_augmentation_factor = 3
            
            # Calculate how many original samples we need to augment
            # and with what factor to meet our target without exceeding the 1:3 ratio
            if deficit <= count * max_augmentation_factor:
                # We can meet the target with a partial augmentation
                samples_to_augment = (deficit + max_augmentation_factor - 1) // max_augmentation_factor
                augmentation_factor = min(max_augmentation_factor, (deficit + samples_to_augment - 1) // samples_to_augment)
            else:
                # We can't meet the target, so augment all samples with the max factor
                samples_to_augment = count
                augmentation_factor = max_augmentation_factor
            
            # Make sure we don't exceed 500 samples after augmentation
            total_after_augmentation = count + (samples_to_augment * augmentation_factor)
            if total_after_augmentation > max_samples_per_age:
                # Adjust the augmentation factor to not exceed the maximum
                deficit = max_samples_per_age - count
                if deficit <= 0:
                    augmentation_factor = 0
                    samples_to_augment = 0
                else:
                    samples_to_augment = min(samples_to_augment, count)
                    augmentation_factor = min(max_augmentation_factor, deficit // samples_to_augment)
                    if augmentation_factor == 0 and deficit > 0:
                        augmentation_factor = 1
                        samples_to_augment = min(deficit, count)
            
            age_augmentation_info[age] = {
                'count': count,
                'augmentation_factor': augmentation_factor,
                'samples_to_augment': samples_to_augment,
                'needs_downsampling': False,
                'keep_count': count
            }
    
    return age_augmentation_info

def create_augmentation_transforms(
    severity: str = "medium"
) -> List[A.Compose]:
    """
    Create a list of Albumentation transforms for different types of augmentations.
    
    Parameters:
    -----------
    severity : str
        The severity level of augmentations: "light", "medium", or "heavy"
        
    Returns:
    --------
    List[A.Compose]
        List of Albumentation transforms to apply
    """
    # Define probability and intensity parameters based on severity
    if severity == "light":
        p_base = 0.3
        brightness_contrast_limit = 0.1
        hue_shift_limit = 5
        sat_shift_limit = 10
        val_shift_limit = 10
        blur_limit = 3
    elif severity == "medium":
        p_base = 0.5
        brightness_contrast_limit = 0.2
        hue_shift_limit = 10
        sat_shift_limit = 20
        val_shift_limit = 20
        blur_limit = 5
    else:  # heavy
        p_base = 0.7
        brightness_contrast_limit = 0.3
        hue_shift_limit = 15
        sat_shift_limit = 30
        val_shift_limit = 30
        blur_limit = 7
    
    # Create a list of transforms
    # These are designed to be realistic for face images, avoiding 
    # distortions that would make recognition difficult
    transforms = [
        # Color-based transforms
        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast_limit,
                contrast_limit=brightness_contrast_limit,
                p=p_base
            ),
            A.HueSaturationValue(
                hue_shift_limit=hue_shift_limit,
                sat_shift_limit=sat_shift_limit,
                val_shift_limit=val_shift_limit,
                p=p_base
            )
        ]),
        
        # Light noise and blur (simulates different camera qualities)
        A.Compose([
            A.GaussNoise(var_limit=(5.0, 20.0), p=p_base),
            A.GaussianBlur(blur_limit=blur_limit, p=p_base)
        ]),
        
        # Lighting changes
        A.Compose([
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1), 
                num_shadows_lower=1, 
                num_shadows_upper=2, 
                shadow_dimension=4, 
                p=p_base
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast_limit,
                contrast_limit=brightness_contrast_limit,
                p=p_base
            )
        ]),
        
        # Quality degradation (simulates compression artifacts)
        A.Compose([
            A.ImageCompression(quality_lower=80, quality_upper=99, p=p_base),
            A.GaussNoise(var_limit=(5.0, 15.0), p=p_base * 0.5)
        ])
    ]
    
    return transforms

def select_samples_for_augmentation(
    df: pd.DataFrame, 
    age_augmentation_info: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Select samples from the dataset for augmentation based on the augmentation info.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the dataset
    age_augmentation_info : Dict[int, Dict[str, Any]]
        Dictionary with age as key and augmentation information as value
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing only the samples selected for augmentation
    """
    samples_to_augment = []
    
    # Group by age
    grouped = df.groupby('age')
    
    # For each age group, select the appropriate number of samples
    for age, group in grouped:
        if age in age_augmentation_info and age_augmentation_info[age]['samples_to_augment'] > 0:
            needed_samples = age_augmentation_info[age]['samples_to_augment']
            
            # If we need all samples from this age group
            if needed_samples >= len(group):
                samples_to_augment.append(group)
            else:
                # Randomly select the needed number of samples
                selected = group.sample(n=needed_samples, random_state=42)
                samples_to_augment.append(selected)
    
    # Combine all selected samples
    if samples_to_augment:
        return pd.concat(samples_to_augment, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns)

def apply_augmentation_to_image(
    image_path: str,
    output_path: str,
    transform: A.Compose
) -> bool:
    """
    Apply a specific augmentation transform to an image and save the result.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_path : str
        Path where the augmented image will be saved
    transform : A.Compose
        Albumentation transform to apply
        
    Returns:
    --------
    bool
        True if augmentation was successful, False otherwise
    """
    try:
        # Read the image
        image = np.array(Image.open(image_path))
        
        # Apply the transform
        augmented = transform(image=image)
        augmented_image = augmented['image']
        
        # Save the augmented image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(augmented_image).save(output_path)
        
        return True
    except Exception as e:
        print(f"Error augmenting {image_path}: {str(e)}")
        return False

def downsample_by_age(
    df: pd.DataFrame,
    age_augmentation_info: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Downsample age groups that exceed the maximum samples per age.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the dataset
    age_augmentation_info : Dict[int, Dict[str, Any]]
        Dictionary with age as key and augmentation/downsampling information as value
        
    Returns:
    --------
    pd.DataFrame
        DataFrame after downsampling
    """
    downsampled_dfs = []
    
    # Group by age
    grouped = df.groupby('age')
    
    # Process each age group
    for age, group in grouped:
        if age in age_augmentation_info:
            info = age_augmentation_info[age]
            if info.get('needs_downsampling', False):
                # Randomly select the specified number of samples
                sampled = group.sample(n=info['keep_count'], random_state=42)
                downsampled_dfs.append(sampled)
            else:
                # Keep all samples for this age
                downsampled_dfs.append(group)
        else:
            # No info for this age, keep all samples
            downsampled_dfs.append(group)
    
    # Combine all processed groups
    return pd.concat(downsampled_dfs, ignore_index=True)

def create_augmented_directories(augmented_base_path: str = "./data/augmented/") -> None:
    """
    Create the necessary directories for augmented images.
    
    Parameters:
    -----------
    augmented_base_path : str
        Base path for storing augmented images
    """
    # Create main augmented directory
    os.makedirs(augmented_base_path, exist_ok=True)
    
    # Create gender-specific subdirectories
    os.makedirs(os.path.join(augmented_base_path, "male"), exist_ok=True)
    os.makedirs(os.path.join(augmented_base_path, "female"), exist_ok=True)
    
    print(f"Created augmented directories at {augmented_base_path}")

def process_and_align_image(
    img_path: str, 
    output_path: str, 
    target_size: tuple = (224, 224),
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 10
) -> Dict[str, Any]:
    """
    Process a single image to extract, align, resize and save the face.
    
    Parameters:
    -----------
    img_path : str
        Path to the source image
    output_path : str
        Path to save the aligned face image
    target_size : tuple
        Size to resize the face image to
    detector_backend : str
        Face detector backend to use
    enforce_detection : bool
        Whether to enforce face detection
    align : bool
        Whether to align the face
    expand_percentage : int
        Percentage to expand the detected face area
        
    Returns:
    --------
    dict
        Dictionary containing information about the processing result:
        - 'success': bool indicating whether processing was successful
        - 'msg': String message about the result
        - 'facial_area': Dictionary containing face region coordinates if successful
        - 'confidence': Face detection confidence score if successful
    """
    result = {
        'success': False,
        'msg': '',
        'facial_area': None,
        'confidence': 0
    }
    
    try:
        # Use DeepFace to extract and align the face
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage
        )
        
        if len(face_objs) == 0:
            result['msg'] = "No face detected in the image"
            return result
        
        # Get the first face (assuming one face per image for this dataset)
        face_obj = face_objs[0]
        aligned_face = face_obj["face"]
        
        # Convert the numpy array to PIL Image
        if isinstance(aligned_face, np.ndarray):
            # If the image is normalized (values between 0 and 1)
            if aligned_face.max() <= 1.0:
                aligned_face = (aligned_face * 255).astype(np.uint8)
            
            # DeepFace extract_faces already returns in RGB format
            face_img = Image.fromarray(aligned_face)
            
            # Resize the image to the target size
            face_img = face_img.resize(target_size, Image.LANCZOS)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the aligned and resized face
            face_img.save(output_path)
            
            result['success'] = True
            result['msg'] = "Face aligned and saved successfully"
            result['facial_area'] = face_obj["facial_area"]
            result['confidence'] = face_obj["confidence"]
        else:
            result['msg'] = "Unexpected format for aligned face"
    
    except Exception as e:
        result['msg'] = f"Error processing image: {str(e)}"
    
    return result

def augment_facial_dataset(
    input_csv_path: str,
    input_base_path: str,
    output_base_path: str,
    output_csv_path: str,
    max_samples_per_age: int = 500,
    severity: str = "medium",
    align_faces: bool = True,
    detector_backend: str = "retinaface",
    target_size: tuple = (224, 224),
    use_cached: bool = False,
    cache_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Augment a facial dataset based on age distribution to ensure balanced representation,
    with no more than the specified maximum samples per age.
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the CSV file containing the dataset information
    input_base_path : str
        Base path of the input image directory
    output_base_path : str
        Base path to save augmented images
    output_csv_path : str
        Path to save the augmented dataset CSV file
    max_samples_per_age : int
        Maximum number of samples allowed per age
    severity : str
        The severity level of augmentations: "light", "medium", or "heavy"
    align_faces : bool
        Whether to align faces before augmentation
    detector_backend : str
        Face detector backend to use for alignment
    target_size : tuple
        Size to resize the face images to
    use_cached : bool
        Whether to use cached processed data if available
    cache_path : Optional[str]
        Path to the cache directory. If None and use_cached is True, 
        a default path will be used
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the balanced dataset information
    """
    # Set default cache path if not provided
    if cache_path is None and use_cached:
        cache_path = os.path.join(os.path.dirname(output_base_path), "cache")
    
    # Check if cached data exists and use_cached is True
    if use_cached and cache_path:
        if os.path.exists(cache_path):
            print(f"Using cached processed data from {cache_path}")
            try:
                cached_df = pd.read_csv(cache_path)
                
                # Verify that all files referenced in the cache exist
                all_files_exist = True
                missing_files = []
                for file_path in cached_df['file_path']:
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        missing_files.append(file_path)
                        if len(missing_files) > 5:  # Limit reporting to the first few files
                            break
                
                if all_files_exist:
                    print(f"Successfully loaded {len(cached_df)} records from cache.")
                    return cached_df
                else:
                    print(f"Warning: {len(missing_files)} files referenced in cache are missing.")
                    print(f"First few missing files: {missing_files[:5]}")
                    print("Will reprocess data.")
            except Exception as e:
                print(f"Error loading cached data: {str(e)}. Will reprocess data.")
    
    # Create output directories
    create_augmented_directories(output_base_path)
    
    # Load the input dataset
    try:
        input_df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(input_df)} records from {input_csv_path}")
    except Exception as e:
        print(f"Error loading the input dataset: {str(e)}")
        raise
    
    # Analyze age distribution
    age_augmentation_info = analyze_age_distribution(
        csv_path=input_csv_path,
        max_samples_per_age=max_samples_per_age
    )
    
    # First, downsample age groups that exceed the maximum
    downsampled_df = downsample_by_age(input_df, age_augmentation_info)
    
    # Print processing plan
    print("\nProcessing plan:")
    for age, info in sorted(age_augmentation_info.items()):
        if info.get('needs_downsampling', False):
            print(f"  Age {age}: {info['count']} samples → downsample to {info['keep_count']} samples")
        elif info['augmentation_factor'] > 0:
            added_samples = info['samples_to_augment'] * info['augmentation_factor']
            target = info['count'] + added_samples
            print(f"  Age {age}: {info['count']} samples → augment {info['samples_to_augment']} samples "
                  f"with factor {info['augmentation_factor']} to add {added_samples} new samples (total: {target})")
        else:
            print(f"  Age {age}: {info['count']} samples → no changes needed")
    
    # Select samples for augmentation
    samples_to_augment = select_samples_for_augmentation(downsampled_df, age_augmentation_info)
    print(f"Selected {len(samples_to_augment)} samples for augmentation")
    
    # Create augmentation transforms
    transforms = create_augmentation_transforms(severity=severity)
    
    # Create a new dataframe for the final dataset
    augmented_rows = []
    
    # Process each sample in the downsampled dataset
    print("Processing original data...")
    for _, row in tqdm(downsampled_df.iterrows(), total=len(downsampled_df), desc="Processing original data"):
        # Define input and output paths
        input_path = row['file_path']
        output_filename = os.path.basename(input_path)
        output_path = os.path.join(output_base_path, row['gender'], output_filename)
        
        # Process the image - either align it or just copy it
        if align_faces:
            result = process_and_align_image(
                img_path=input_path,
                output_path=output_path,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
                expand_percentage=10
            )
            success = result['success']
        else:
            # Just copy the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                shutil.copy2(input_path, output_path)
                success = True
            except Exception as e:
                print(f"Error copying {input_path}: {str(e)}")
                success = False
        
        # Add to final dataframe if successful
        if success:
            new_row = row.copy()
            new_row['file_path'] = output_path
            augmented_rows.append(new_row)
    
    # Perform augmentation
    print("Generating augmented data...")
    for _, row in tqdm(samples_to_augment.iterrows(), total=len(samples_to_augment), desc="Augmenting samples"):
        age = row['age']
        augmentation_factor = age_augmentation_info[age]['augmentation_factor']
        
        # Skip if no augmentation needed
        if augmentation_factor <= 0:
            continue
        
        # Get the input path for augmentations (use the aligned/processed image)
        input_path = os.path.join(output_base_path, row['gender'], os.path.basename(row['file_path']))
        
        # Parse the original filename to extract name, gender, and age components
        # Expected format: "Name_gender_age.ext"
        filename = os.path.basename(row['file_path'])
        name_parts = os.path.splitext(filename)[0].split('_')
        ext = os.path.splitext(filename)[1]
        
        # Generate augmentations
        for i in range(augmentation_factor):
            # Select a random transform
            transform = random.choice(transforms)
            
            # Define output path based on filename pattern
            if len(name_parts) >= 3:
                # Extract components
                name = name_parts[0]
                gender_age_parts = '_'.join(name_parts[1:])  # Keep the rest of the filename intact
                aug_filename = f"{name}-{i+1}_{gender_age_parts}{ext}"
            else:
                # Simple fallback if filename doesn't match expected pattern
                base_name = os.path.splitext(filename)[0]
                aug_filename = f"{base_name}-{i+1}{ext}"
            
            aug_path = os.path.join(output_base_path, row['gender'], aug_filename)
            
            # Apply the augmentation
            success = apply_augmentation_to_image(
                image_path=input_path,
                output_path=aug_path,
                transform=transform
            )
            
            # Add to final dataframe if successful
            if success:
                new_row = row.copy()
                # Update the name in the DataFrame to match the file naming convention
                new_row['name'] = f"{row['name']}-{i+1}"
                new_row['file_name'] = aug_filename
                new_row['file_path'] = aug_path
                augmented_rows.append(new_row)
    
    # Create the final dataset dataframe
    final_df = pd.DataFrame(augmented_rows)
    
    # Check if we have too many samples for any age after augmentation
    age_counts = final_df['age'].value_counts()
    ages_over_limit = [age for age, count in age_counts.items() if count > max_samples_per_age]
    
    # If any age has too many samples, downsample again
    if ages_over_limit:
        print(f"\nWarning: After augmentation, {len(ages_over_limit)} age groups still exceed {max_samples_per_age} samples.")
        print("Performing final downsample to enforce maximum samples per age...")
        
        # Create a new age_info dictionary for the final downsampling
        final_downsample_info = {
            age: {
                'count': count,
                'needs_downsampling': True,
                'keep_count': max_samples_per_age
            } for age, count in age_counts.items() if count > max_samples_per_age
        }
        
        # Downsample the final dataset
        final_df = downsample_by_age(final_df, final_downsample_info)
    
    # Save the final dataset
    try:
        final_df.to_csv(output_csv_path, index=False)
        print(f"Final dataset saved to {output_csv_path}")
        
        # Cache the processed data for future use
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            final_df.to_csv(cache_path, index=False)
            print(f"Cached processed data saved to {cache_path}")
            
            # Save additional metadata for cache validation
            metadata = {
                "created_timestamp": pd.Timestamp.now().isoformat(),
                "input_csv": input_csv_path,
                "original_record_count": len(input_df),
                "final_record_count": len(final_df),
                "parameters": {
                    "max_samples_per_age": max_samples_per_age,
                    "severity": severity,
                    "align_faces": align_faces,
                    "target_size": str(target_size)
                }
            }
            
            with open(os.path.join(cache_path, "metadata.json"), "w") as f:
                import json
                json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error saving final dataset: {str(e)}")
    
    # Print final statistics
    final_age_counts = final_df['age'].value_counts().to_dict()
    print("\nFinal age distribution:")
    for age in sorted(final_age_counts.keys()):
        original_count = age_augmentation_info[age]['count'] if age in age_augmentation_info else 0
        print(f"  Age {age}: {original_count} → {final_age_counts[age]} samples")
    
    return final_df