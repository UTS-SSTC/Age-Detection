import torch
import torch.nn as nn

from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from .models import load_efficientnet_b0, get_device


def compute_loss(model, data_loader, device, criterion):
    """
    Compute model loss on validation dataset.

    Parameters:
    -----------
    model: nn.Module
        Model to validate
    data_loader: torch.utils.data.DataLoader
        Data loader
    device: torch.device
        Device to use
    age_criterion: function
        Loss function for age prediction

    Returns:
    --------
    float
        Mean validation loss across all batches
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)


class FineTunedBackbone(nn.Module):
    """
    Adaptable backbone network for transfer learning with partial parameter freezing fine-tuning.

    Parameters:
    -----------
    pretrained_model : torch.nn.Module
        Pretrained backbone model
    backbone_fn : function
        Backbone constructor function
    pretrained : bool
        Whether to load pretrained weights
    train_model : bool
        Whether to set the model to training mode
    freeze_rate : float
        The proportion of frozen parameters
    """

    def __init__(self, pretrained_model=None, backbone_fn=load_efficientnet_b0,
                 pretrained=True, train_model=False, freeze_rate=0.8):
        super().__init__()
        # Loading pre-trained backbone network
        if pretrained_model is None:
            pretrained_model = backbone_fn(
                pretrained=pretrained,
                train_mode=train_model,
                device=get_device()
            )

        self.backbone = nn.Sequential(
            # Remove the last two modules (classification header)
            *list(pretrained_model.children())[:-2],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten()
        )

        # Freeze part of the parameters
        total_layers = len(list(self.backbone.children()))
        freeze_idx = int(total_layers * freeze_rate)
        for idx, (name, param) in enumerate(self.backbone.named_parameters()):
            if idx < freeze_idx:
                param.requires_grad = False

        # Add a regression header
        self.age_head = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass of backbone.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        features = self.backbone(x)
        age_output = self.age_head(features)
        return age_output.squeeze()

    def finetune(self, train_loader, val_loader, epochs=10, lr=1e-4):
        """
        Fine-tune backbone model

        Parameters:
        -----------
        train_loader: torch.utils.data.DataLoader
            Training data loader
        val_loader: torch.utils.data.DataLoader
            Validation data loader
        epochs: int
            Number of epochs
        lr: float
            Learning rate
        """
        device = get_device()
        self.to(device)

        # Define the loss
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # Cosine annealing scheduler
        total_steps = epochs * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f'Fine tuning Epoch {epoch + 1}'):
                images = batch[0].to(device)
                labels = batch[1].to(device)

                optimizer.zero_grad()
                pred = self(images)
                loss = criterion(pred, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            val_loss = compute_loss(self, val_loader, device, criterion)

            print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")

        # Save feature extractor parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
