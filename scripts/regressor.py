import pickle

import torch
import lightgbm
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from .models import load_efficientnet_b0, get_device


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
        # Directly load the pretrained model
        backbone = load_efficientnet_b0(
            pretrained=True,
            train_mode=False,
            device=get_device()
        )

        return nn.Sequential(
            backbone,
            nn.Flatten()
        )
    else:
        # Use fine-tuned backbone network
        return nn.Sequential(
            model.backbone,
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
                batch_features = torch.flatten(batch_features, start_dim=1).cpu().numpy()

                features.append(batch_features)
                ages.append(batch[1].numpy())

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

    def save_model(self, path):
        """
        Saves the model components to a single file.

        Parameters:
        -----------
        path : str
            Path to save model
        """
        # Save PyTorch feature extractor parameters
        feature_extractor_state = self.feature_extractor.state_dict()

        # Serialize LightGBM model as a byte stream
        gbm_bytes = pickle.dumps(self.gbm)

        # Save to checkpoint file uniformly
        torch.save({
            'feature_extractor_state': feature_extractor_state,
            'gbm_bytes': gbm_bytes
        }, path)

    def load_model(self, path):
        """
        Loads model components from a saved file.

        Parameters:
        -----------
        path : str
            Path to load model
        """
        # Load the checkpoint file to the current device
        checkpoint = torch.load(path, map_location=self.device)

        # Recover Feature Extractor
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])

        # Deserialize LightGBM model
        self.gbm = pickle.loads(checkpoint['gbm_bytes'])
