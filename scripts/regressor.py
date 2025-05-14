import pickle

import torch
import joblib
import lightgbm
from collections import OrderedDict
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from .models import load_efficientnet_b0, get_device
from .backbone import FineTunedBackbone


def build_feature_extractor(model):
    """
    Constructs a feature extractor from either a pretrained or fine-tuned model.

    Parameters:
    -----------
    model: torch.nn.Module
        Base model for feature extraction
        If None, uses pretrained EfficientNet-B0 with frozen parameters

    Returns:
    --------
    torch.nn.Sequential
        Feature extractor
    """
    if model is None:
        model = FineTunedBackbone()

    return nn.Sequential(
        model.backbone,
        model.fc,
        nn.Flatten()
    )


class DeepFeatureLGBM(nn.Module):
    """
    Hybrid deep learning/lightGBM regressor model for end-to-end feature extraction and regression.

    Parameters:
    -----------
    fineTuned_model : torch.nn.Module
        Fine-tuned model
    light_gbm_config : dictionary
        Configuration parameters for LightGBM
    """

    def __init__(self, fine_tuned_model=None, light_gbm_config=None):
        super(DeepFeatureLGBM, self).__init__()
        self.device = get_device()
        # Feature extraction stage
        self.feature_extractor = build_feature_extractor(fine_tuned_model)

        # LightGBM model
        if light_gbm_config is None:
            light_gbm_config = {
                'objective': 'regression',
                'num_leaves': 40,
                'learning_rate': 0.05,
            }
        self.gbm = lightgbm.LGBMRegressor(**light_gbm_config)

    def forward(self, x):
        """
        Standard forward propagation (for feature extraction).

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (batch_size, channels, height, width)

        Returns:
        --------
        numpy.ndarray
            Flattened feature vectors (batch_size, num_features)
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.cpu().numpy()

    def train_gbm(self, train_loader):
        """
        Train LightGBM regression head using extracted features.

        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            Training data loader
        """
        # Extract features and labels
        features, labels = self._extract_features(train_loader)

        # Training regression model
        self.gbm.fit(features, labels, callbacks=[lightgbm.log_evaluation(period=10)])

    def _extract_features(self, dataloader):
        """
        Batch-wise feature extraction.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Training data loader

        Returns:
        --------
        tuple
            Features and labels
        """
        self.eval()
        features = []
        ages = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting features'):
                images = batch[0].to(self.device)

                batch_features = self.feature_extractor(images)
                # batch_features = torch.flatten(batch_features, start_dim=1).cpu().numpy()
                batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()

                features.append(batch_features)
                ages.append(batch[1].numpy())
                # print(f"Batch features shape before flatten: {batch_features.shape}")
                # break

        return np.concatenate(features, axis=0), np.concatenate(ages, axis=0)

    def predict(self, dataloader):
        """
        End-to-end prediction

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Training data loader

        Returns:
        --------
        numpy.ndarray
            Model predictions (batch_size,)
        """
        X, _ = self._extract_features(dataloader)

        return self.gbm.predict(X)

    def save_model(self, path, name):
        """
        Saves the model components to a single file.

        Parameters:
        -----------
        path : str
            Path to save model
        name : str
            Name of saved model
        """
        torch.save(self.feature_extractor.state_dict(), f"{path}/{name}.pth")

        joblib.dump(self.gbm, f"{path}/{name}.pkl")

    def load_model(self, path, name):
        """
        Loads model components from a saved file.

        Parameters:
        -----------
        path : str
            Path to load model
        name : str
            Name of saved model
        """
        def remove_module_prefix(state_dict):
            """
            Strip the 'module.' prefix from state_dict keys (if present).

            When a model is wrapped in torch.nn.DataParallel or DistributedDataParallel,
            each key is prefixed with 'module. This helper removes that prefix so keys match the original model.

            Parameters:
            -----------
            state_dict : dict
            The original state_dict loaded from file, which may contain
            keys like 'module.layer.weight'.

            Returns:
            --------
            OrderedDict
            A new state_dict with 'module.' removed from any keys.
            """
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            return new_state_dict

        state_dict = torch.load(f"{path}/{name}.pth", map_location=self.device)
        state_dict = remove_module_prefix(state_dict)
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.to(self.device).eval()

        self.gbm = joblib.load(f"{path}/{name}.pkl")
