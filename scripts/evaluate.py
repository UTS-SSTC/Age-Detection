import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import scripts.preprocessing as sp
import matplotlib.pyplot as plt

class AgeDataset(Dataset):
    """
    Dataset class for loading age prediction images and corresponding labels.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing image file paths and age labels
    transform : torchvision.transforms
        Image transformations to apply
    root_dir : str
        Root directory of the image dataset
    """
    def __init__(self, metadata_df, transform=None, root_dir="./data/processed/"):
        self.metadata_df = metadata_df
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample to get
            
        Returns:
        --------
        tuple
            (image, age) pair where image is the transformed image tensor
            and age is the target age label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get file path and age from metadata
        row = self.metadata_df.iloc[idx]
        img_path = row['file_path']
        age = row['age']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder black image if the file can't be loaded
            image = Image.new('RGB', (224, 224), color='black')
            
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32)

def create_data_transforms(input_size=224):
    """
    Create standard data transformations for train and evaluation sets.
    
    Parameters:
    -----------
    input_size : int
        Size (in pixels) to resize images to
        
    Returns:
    --------
    dict
        Dictionary containing 'train', 'val', and 'test' transforms
    """
    # Define normalization parameters (ImageNet standard)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    
    return data_transforms

def load_age_data(metadata_path, train_ratio=0.7, val_ratio=0.15, 
                  batch_size=32, input_size=224, num_workers=4, use_cached=True,
                  wiki_mat_path=None, verified_image_path=None, min_age=0, max_age=100):
    """
    Load age dataset from metadata and create train/val/test DataLoaders.
    
    Parameters:
    -----------
    metadata_path : str
        Path to the metadata CSV file
    train_ratio : float
        Proportion of data to use for training
    val_ratio : float
        Proportion of data to use for validation
    test_ratio : float
        Proportion of data to use for testing
    batch_size : int
        Batch size for DataLoaders
    input_size : int
        Size (in pixels) to resize images to
    num_workers : int
        Number of worker threads for data loading
    use_cached : bool
        Whether to use cached metadata
    wiki_mat_path : str
        Path to wiki.mat file (used if metadata needs to be generated)
    verified_image_path : str
        Path to verified images (used if metadata needs to be generated)
    min_age : int
        Minimum age to include
    max_age : int
        Maximum age to include
        
    Returns:
    --------
    dict
        Dictionary containing DataLoaders and metadata
    """
    # Load or process metadata
    if use_cached and os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded cached metadata from {metadata_path}")
    else:
        if wiki_mat_path is None or verified_image_path is None:
            raise ValueError("When not using cached metadata, wiki_mat_path and verified_image_path must be provided")
        
        # Process metadata using the preprocessing module
        metadata_df = sp.process_wiki_metadata(
            wiki_mat_path, 
            min_age=min_age, 
            max_age=max_age, 
            verified_image_path=verified_image_path,
            use_cached=False,
            cache_path=metadata_path,
            copy_images=False  # Avoid copying images during evaluation
        )
    
    print(f"Dataset contains {len(metadata_df)} samples")
    
    # Create data transformations
    data_transforms = create_data_transforms(input_size)
    
    # Shuffle the dataset
    metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train, val, test sets
    train_size = int(train_ratio * len(metadata_df))
    val_size = int(val_ratio * len(metadata_df))
    
    train_df = metadata_df[:train_size]
    val_df = metadata_df[train_size:train_size+val_size]
    test_df = metadata_df[train_size+val_size:]
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = AgeDataset(train_df, transform=data_transforms['train'])
    val_dataset = AgeDataset(val_df, transform=data_transforms['val'])
    test_dataset = AgeDataset(test_df, transform=data_transforms['test'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Return everything in a dictionary
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'metadata_df': metadata_df
    }


def compute_accuracy(predictions, labels, plot=False):
    """
    Compute exact match accuracy (after rounding predictions).
    Optionally plots prediction vs ground truth distribution.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to display a histogram comparison (default: False).

    Returns
    -------
    float
        Accuracy score in range [0, 1].
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).sum().item()
    accuracy = correct / len(labels)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(labels.numpy(), bins=range(int(labels.min()), int(labels.max()) + 2),
                 alpha=0.6, label='Ground Truth', edgecolor='black')
        plt.hist(rounded_preds.numpy(), bins=range(int(labels.min()), int(labels.max()) + 2),
                 alpha=0.6, label='Predictions (Rounded)', edgecolor='black')
        plt.title(f"Prediction vs Ground Truth (Accuracy = {accuracy:.4f})")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return accuracy

def compute_loss(predictions, labels, loss_fn=torch.nn.MSELoss(), plot=False):
    """
    Compute scalar loss value and optionally visualize residual distribution.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    loss_fn : callable
        PyTorch loss function (default: MSELoss).
    plot : bool, optional
        Whether to plot the error distribution (default: False).

    Returns
    -------
    float
        Scalar loss value.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    loss = loss_fn(predictions, labels)

    if plot:
        residuals = (predictions - labels).numpy()
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=30, color='lightsalmon', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"Residual Distribution (Loss = {loss.item():.4f})")
        plt.xlabel("Residual (Prediction - Ground Truth)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return loss.item()

def compute_mae(predictions, labels, plot=False):
    """
    Compute Mean Absolute Error (MAE) and optionally plot error distribution.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to plot error histogram (default: False).

    Returns
    -------
    float
        MAE value.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    residuals = torch.abs(predictions - labels)
    mae = torch.mean(residuals)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(residuals.numpy(), bins=30, color='orange', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"Absolute Error Distribution (MAE = {mae.item():.2f})")
        plt.xlabel("Absolute Error |Prediction - Ground Truth|")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mae.item()

def compute_mse(predictions, labels, plot=False):
    """
    Compute Mean Squared Error (MSE) and optionally plot residual distribution.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to show a residual histogram (default: False).

    Returns
    -------
    float
        MSE value.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    residuals = predictions - labels
    mse = torch.mean(residuals ** 2)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(residuals.numpy(), bins=30, color='lightcoral', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"Residual Distribution (MSE = {mse.item():.2f})")
        plt.xlabel("Residual (Prediction - Ground Truth)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mse.item()

def compute_rmse(predictions, labels, plot=False):
    """
    Compute Root Mean Squared Error (RMSE) and optionally visualize residuals.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to plot residuals (default: False).

    Returns
    -------
    float
        RMSE value.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    residuals = predictions - labels
    rmse = torch.sqrt(torch.mean(residuals ** 2))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(residuals.numpy(), bins=30, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"Residual Distribution (RMSE = {rmse.item():.2f})")
        plt.xlabel("Residual (Prediction - Ground Truth)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rmse.item()

def compute_r2(predictions, labels, plot=False):
    """
    Compute R² score and optionally plot predicted vs true values.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to plot scatter of predictions vs labels.

    Returns
    -------
    float
        R² score in range (-∞, 1].
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    ss_res = torch.sum((labels - predictions) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(labels.numpy(), predictions.numpy(), alpha=0.6, color='blue', edgecolors='k')
        plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', label='Ideal Fit')
        plt.xlabel("Ground Truth (True Age)")
        plt.ylabel("Predicted Age")
        plt.title(f"R² = {r2.item():.4f}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return r2.item()

def compute_five_off_accuracy(predictions, labels, plot=False):
    """
    Compute ±5 error tolerance accuracy and optionally plot error distribution.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool, optional
        Whether to plot absolute error distribution (default: False).

    Returns
    -------
    float
        5-off accuracy score.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    errors = torch.abs(predictions - labels)
    accuracy = (errors <= 5.0).sum().item() / len(labels)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(errors.numpy(), bins=30, color='lightgreen', edgecolor='black')
        plt.axvline(x=5.0, color='red', linestyle='--', label='±5 Threshold')
        plt.title("Prediction Error Distribution (5-off Accuracy)")
        plt.xlabel("Absolute Error (|Prediction - Label|)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return accuracy

def plot_uncertainty_distribution(predictions, labels, bins=30):
    """
    Plot distribution of residuals (prediction error) for uncertainty visualization.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    bins : int, optional
        Number of histogram bins (default: 30).

    Returns
    -------
    None
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()
    residuals = predictions - labels  # 预测误差（残差）

    plt.figure(figsize=(8, 5))
    plt.hist(residuals.numpy(), bins=bins, color='lightblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect Prediction (Residual=0)')
    plt.title("Uncertainty Distribution")
    plt.xlabel("Residual = Prediction - Ground Truth")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
