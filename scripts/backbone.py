import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from .models import load_efficientnet_b0, get_device


class FineTunedBackbone(nn.Module):
    """
    Adaptable backbone network for transfer learning with Deep Label Distribution Learning (DLDL) for age regression.

    Parameters:
    -----------
    pretrained_model : torch.nn.Module
        Pretrained backbone model
    backbone_fn : function
        Backbone constructor function
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    freeze_rate : float
        The proportion of frozen parameters in the backbone
    age_min : int
        Minimum age in label distribution (inclusive)
    age_max : int
        Maximum age in label distribution (inclusive)
    sigma : float
        Standard deviation for Gaussian label distribution
    lambda_reg : float
        Weight for regression loss term
    """

    def __init__(
            self,
            pretrained_model=None,
            backbone_fn=load_efficientnet_b0,
            pretrained=True,
            train_mode=False,
            freeze_rate=0.8,
            age_min=0,
            age_max=100,
            sigma=3.0,
            lambda_reg=0.5
    ):
        super().__init__()
        # Load pre-trained backbone network
        if pretrained_model is None:
            pretrained_model = backbone_fn(
                pretrained=pretrained,
                train_mode=train_mode,
                device=get_device()
            )

        # Feature extractor: remove classifier head
        self.backbone = nn.Sequential(
            *list(pretrained_model.children())[:-2],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten()
        )

        # Freeze a portion of backbone parameters
        total_params = sum(1 for _ in self.backbone.parameters())
        freeze_count = int(total_params * freeze_rate)
        for idx, param in enumerate(self.backbone.parameters()):
            if idx < freeze_count:
                param.requires_grad = False

        # Age distribution settings
        self.age_min = age_min
        self.age_max = age_max
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.num_classes = age_max - age_min + 1

        # Register age values buffer for expectation
        self.register_buffer('age_values', torch.arange(age_min, age_max + 1, dtype=torch.float))

        # Regression head outputs distribution logits
        dummy_input = torch.randn(1, 3, 224, 224).to(get_device())
        with torch.no_grad():
            feat_dim = self.backbone(dummy_input).size(1)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.age_head = nn.Linear(128, self.num_classes)

    def _make_label_dist(self, ages):
        """
        Construct Gaussian label distributions for a batch of ages.

        Parameters:
        -----------
        ages: torch.Tensor
            Batch of ages
        Returns:
        --------
        torch.Tensor
            Gaussian label distribution
        """
        # ages shape: (B,)
        # age_values shape: (num_classes,)
        diffs = (self.age_values.unsqueeze(0) - ages.unsqueeze(1)) ** 2
        gauss = torch.exp(-diffs / (2 * self.sigma ** 2))
        return gauss / gauss.sum(dim=1, keepdim=True)

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
        x = self.backbone(x)
        features = self.fc(x)
        logits = self.age_head(features)
        probs = F.softmax(logits, dim=1)
        ages_pred = (probs * self.age_values).sum(dim=1)
        return ages_pred

    def finetune(self, train_loader, val_loader, epochs=10, lr=3e-4, preheat_epochs=5, preheat_lr=1e-3):
        """
        Fine-tune backbone model using DLDL

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
        preheat_epochs: int
            Number of preheat epochs
        preheat_lr: float
            Learning rate for preheat epochs
        """
        device = get_device()
        self.to(device)

        # Define losses and optimizer
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_reg = nn.L1Loss()

        # Phase 1: Pretrain the age head
        if preheat_epochs > 0:
            # Ensure only the age head is trainable
            original_trainables = []
            for param in self.parameters():
                original_trainables.append(param.requires_grad)
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = True
            for param in self.age_head.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=preheat_lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=preheat_epochs * len(train_loader), eta_min=1e-6)

            for epoch in range(preheat_epochs):
                self.train()
                total_loss = 0.0
                for batch in tqdm(train_loader, desc=f'Pretrain Head Epoch {epoch + 1}'):
                    images, labels = batch[0].to(device), batch[1].to(device)

                    dist = self._make_label_dist(labels)

                    optimizer.zero_grad()
                    features = self.backbone(images)
                    features = self.fc(features)
                    logits = self.age_head(features)
                    log_probs = F.log_softmax(logits, dim=1)

                    loss_k1 = criterion_kl(log_probs, dist)
                    probs = torch.exp(log_probs)
                    pred_ages = (probs * self.age_values.to(device)).sum(dim=1)
                    loss_reg = criterion_reg(pred_ages, labels)
                    loss = loss_k1 + loss_reg

                    loss.backward()
                    optimizer.step()

                    scheduler.step()
                    total_loss += loss.item()

                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)

                        pred_ages = self(images)
                        val_loss += criterion_reg(pred_ages, labels).item()
                val_loss /= len(val_loader)
                print(f"Pretrain Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

            # Restore original parameter trainability
            for param, original in zip(self.parameters(), original_trainables):
                param.requires_grad = original

        total_steps = epochs * len(train_loader)
        # Group learning rate
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if 'age_head' in name or 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=5e-2)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=False
        )

        for epoch in range(epochs):
            # Training
            self.train()
            total_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):  # labels: (B,)
                images = images.to(device)
                labels = labels.to(device).float()

                # Build label distribution
                dist = self._make_label_dist(labels)

                optimizer.zero_grad()
                # Manual forward to get logits
                feats = self.backbone(images)
                feats = self.fc(feats)
                logits = self.age_head(feats)
                log_probs = F.log_softmax(logits, dim=1)

                # Losses
                loss_kl = criterion_kl(log_probs, dist)
                probs = torch.exp(log_probs)
                pred_ages = (probs * self.age_values.to(device)).sum(dim=1)
                loss_reg = criterion_reg(pred_ages, labels)
                loss = loss_kl + self.lambda_reg * loss_reg

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).float()
                    # Forward
                    pred_ages = self(images)
                    val_loss += criterion_reg(pred_ages, labels).item()
            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1}: Train Loss={total_loss / len(train_loader):.4f}, Val MAE={val_loss:.4f}')

        # Freeze backbone after fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = False
