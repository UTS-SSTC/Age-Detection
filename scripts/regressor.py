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


class EfficientLightGBM(nn.Module):
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
        super(EfficientLightGBM, self).__init__()
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
        device = get_device()

        # Extract features and labels
        features, labels = self._extract_features(train_loader, device)

        # Training regression model
        self.gbm.fit(features, labels, callbacks=[lightgbm.log_evaluation(period=10)])

    def _extract_features(self, dataloader, device):
        """
        Batch-wise feature extraction.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            Training data loader
        device : torch.device
            Device used for training (GPU or CPU)

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
                images = batch[0].to(device)

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
        device = get_device()
        X, _ = self._extract_features(dataloader, device)

        return self.gbm.predict(X)

    def save_model(self, path):
        """
        Saves the model components to a single file.

        Parameters:
        -----------
        path : str
            Path to save model
        """
        if not hasattr(self.gbm, 'booster_') or self.gbm.booster_ is None:
            raise RuntimeError("LightGBM model not trained. Call train_gbm() before saving.")

        state = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'lgbm_params': self.gbm.get_params(),
            'lgbm_model': self.gbm.booster_.model_to_string(),
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Loads model components from a saved file.

        Parameters:
        -----------
        path : str
            Path to load model
        """
        device = get_device()
        state = torch.load(path, map_location=device)

        # Feature Extractor Parameter Adaptation
        src_params = state['feature_extractor']
        target_params = self.feature_extractor.state_dict()

        # Automatically match different named parameters
        matched_params = {}
        for target_key in target_params:
            possible_src_keys = [
                target_key,
                target_key.replace("stem", "0").replace(".conv", ".0.conv"),
                target_key.replace("stem", "0").replace(".bn", ".1.bn")
            ]

            for src_key in possible_src_keys:
                if src_key in src_params:
                    matched_params[target_key] = src_params[src_key]
                    break
            else:
                raise RuntimeError(f"Parameter {target_key} not found in saved model")

        self.feature_extractor.load_state_dict(matched_params)
        self.feature_extractor.to(device)

        # Initialize and load the LightGBM model
        self.gbm = lightgbm.LGBMRegressor(**state['lgbm_params'])
        self.gbm._Booster = lightgbm.Booster(model_str=state['lgbm_model'])
        self.gbm._n_features = self.gbm._Booster.num_feature()

        # Set sklearn compatibility properties
        self.gbm.fitted_ = True
        self.gbm._n_classes = 1
        self.gbm._classes_ = np.array([0])
