import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import scripts.preprocessing as sp
import matplotlib.pyplot as plt
import numpy as np
import scripts.models as sm

class AgeDataset(Dataset):
    """
    PyTorch Dataset for age prediction.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        DataFrame containing image file paths and corresponding age labels.
    transform : torchvision.transforms.Compose, optional
        Transformations to apply to the images.
    """
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples.

        Returns:
        --------
        int
            Number of samples.
        """
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Fetch a single sample by index.

        Parameters:
        -----------
        idx : int
            Index of the sample.

        Returns:
        --------
        tuple
            A tuple of (image_tensor, age_label).
        """
        row = self.metadata_df.iloc[idx]
        img_path = row['file_path']
        age = row['age']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]

    return {
        'train': transforms.Compose(train_augmentations + base_transforms),
        'val': transforms.Compose(base_transforms),
        'test': transforms.Compose(base_transforms)
    }

def load_age_data(metadata_path, train_ratio=0.7, val_ratio=0.15,
                  batch_size=32, input_size=224, num_workers=4, use_cached=True,
                  wiki_mat_path=None, verified_image_path=None, min_age=0, max_age=100):
    """
    Load and split the Wiki age dataset into train, validation, and test sets.

    Parameters:
    -----------
    metadata_path : str
        Path to cached or generated metadata CSV.
    train_ratio : float
        Ratio of data to allocate to training.
    val_ratio : float
        Ratio of data to allocate to validation.
    batch_size : int
        Number of samples per batch.
    input_size : int
        Image resize dimensions (height and width).
    num_workers : int
        Number of subprocesses to use for data loading.
    use_cached : bool
        If True, load metadata from cache. Otherwise, generate from raw .mat.
    wiki_mat_path : str, optional
        Path to the .mat file (required if not using cache).
    verified_image_path : str, optional
        Path to verified images (required if not using cache).
    min_age : int
        Minimum age to filter.
    max_age : int
        Maximum age to filter.

    Returns:
    --------
    dict
        Dictionary with keys: 'train_loader', 'val_loader', 'test_loader', 'train_df', 'val_df', 'test_df', 'metadata_df'.
    """
    # Load or process metadata
    if use_cached and os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        print(f"[✓] Loaded cached metadata from {metadata_path}")
    else:
        if wiki_mat_path is None or verified_image_path is None:
            raise ValueError("wiki_mat_path and verified_image_path are required if not using cached metadata")

        # Process metadata using the preprocessing module
        metadata_df = sp.process_wiki_metadata(
            mat_file_path=wiki_mat_path,
            min_age=min_age,
            max_age=max_age,
            verified_image_path=verified_image_path,
            use_cached=False,
            cache_path=metadata_path,
            copy_images=False
        )

        print(f"[→] Total samples: {len(metadata_df)}")
    
    # Create data transformations
    data_transforms = create_data_transforms(input_size)
    
    # Shuffle the dataset
    metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train, val, test sets
    train_size = int(train_ratio * len(metadata_df))
    val_size = int(val_ratio * len(metadata_df))
    test_size = len(metadata_df) - train_size - val_size

    train_df = metadata_df[:train_size]
    val_df = metadata_df[train_size:train_size + val_size]
    test_df = metadata_df[train_size + val_size:]

    print(f"[✓] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create dataloaders
    train_loader = DataLoader(AgeDataset(train_df, data_transforms['train']), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(AgeDataset(val_df, data_transforms['val']), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(AgeDataset(test_df, data_transforms['test']), batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

def compute_three_off_accuracy(predictions, labels, plot=False):
    """
    Compute ±3 error tolerance accuracy and optionally plot error distribution.

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
        Accuracy score within ±3 error range.
    """
    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    errors = torch.abs(predictions - labels)
    within_3 = (errors <= 3.0).sum().item()
    accuracy = within_3 / len(labels)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(errors.numpy(), bins=30, color='lightgreen', edgecolor='black')
        plt.axvline(x=3.0, color='red', linestyle='--', label='±3 Threshold')
        plt.title("Prediction Error Distribution (±3 Accuracy)")
        plt.xlabel("Absolute Error (|Prediction - Label|)")
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

def extract_preds_and_labels(model, dataloader, device):
    """
    Extract all predictions and labels for a given model and dataloader.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds.append(out.cpu())
            labels.append(labs.cpu())
    return torch.cat(preds), torch.cat(labels)


def evaluate_all(model, dataloader, device, plot=True):
    """
    Full evaluation: extract preds/labels then compute all metrics and optionally plot.
    """
    preds, labels = extract_preds_and_labels(model, dataloader, device)
    return evaluate_metrics(preds, labels, plot=plot)


def evaluate_metrics(predictions, labels, plot=False):
    """
    Wrapper to call all seven evaluation functions:
      - compute_three_off_accuracy
      - compute_loss
      - compute_mae
      - compute_mse
      - compute_rmse
      - compute_r2
      - compute_five_off_accuracy

    Parameters:
    -----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    plot : bool
        Whether to enable each function's plotting.

    Returns:
    --------
    dict
        Dictionary of metric names and their computed values.
    """
    results = {
        'ThreeOffAccuracy': compute_three_off_accuracy(predictions, labels, plot=False),
        'Loss': compute_loss(predictions, labels, plot=False),
        'MAE': compute_mae(predictions, labels, plot=False),
        'MSE': compute_mse(predictions, labels, plot=False),
        'RMSE': compute_rmse(predictions, labels, plot=False),
        'R2': compute_r2(predictions, labels, plot=False),
        'FiveOffAcc': compute_five_off_accuracy(predictions, labels, plot=False)
    }

    if plot:
        compute_three_off_accuracy(predictions, labels, plot=True)
        compute_loss(predictions, labels, plot=True)
        compute_mae(predictions, labels, plot=True)
        compute_mse(predictions, labels, plot=True)
        compute_rmse(predictions, labels, plot=True)
        compute_r2(predictions, labels, plot=True)
        compute_five_off_accuracy(predictions, labels, plot=True)

    return results


def compare_models(models, dataloader, device, model_names=None, show_plot=True):
    """
    Compare multiple PyTorch models based on evaluation metrics.

    Parameters
    ----------
    models : list
        A list of PyTorch model instances to be evaluated.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the dataset for evaluation.
    device : torch.device
        The device (CPU or GPU) on which the models are evaluated.
    model_names : list of str, optional
        Custom names for the models. If None, default names will be assigned.
    show_plot : bool, optional
        If True, display a bar chart comparing the models' performance.

    Returns
    -------
    dict
        A dictionary where keys are model names and values are dictionaries of evaluation metrics.
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(models))]

    results = {}
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_all(model, dataloader, device, plot=False)

    print(f"\n{'Metric':<15} " + " ".join(f"{n:<15}" for n in model_names))
    for metric in results[model_names[0]]:
        line = f"{metric:<15} " + " ".join(f"{results[m][metric]:<15.4f}" for m in model_names)
        print(line)

    if show_plot:
        x = np.arange(len(results[model_names[0]]))
        width = 0.8 / len(models)
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(model_names):
            vals = [results[name][m] for m in results[name]]
            plt.bar(x + i * width, vals, width, label=name)
        plt.xticks(x + width * (len(models) - 1) / 2, list(results[model_names[0]].keys()), rotation=45)
        plt.ylabel("Metric")
        plt.title("Model Comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results



def evaluate_efficient_lgbm(model, dataloader):
    """
    Evaluate an EfficientLightGBM model using the same evaluation metrics.

    Parameters:
    -----------
    model : EfficientLightGBM
        Trained EfficientLightGBM model
    dataloader : DataLoader
        DataLoader containing image and age labels

    Returns:
    --------
    dict
        Evaluation metrics
    """
    device=sm.get_device()
    preds = torch.tensor(model.predict(dataloader), dtype=torch.float32)
    labels = torch.tensor(np.concatenate([b[1].numpy() for b in dataloader]), dtype=torch.float32)
    return evaluate_metrics(preds, labels, plot=True)
