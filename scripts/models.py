import torch
from torchinfo import summary

def get_device():
    """
    Determine and return the device (CPU or CUDA) to use for training models.
    
    Returns:
    --------
    torch.device
        Device to use for training (CPU or CUDA)
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_efficientnet_b0(pretrained=True, train_mode=True, device=None):
    """
    Load the EfficientNet-B0 model.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    device : torch.device
        Device to load the model on (if None, determined automatically)
        
    Returns:
    --------
    torch.nn.Module
        Loaded EfficientNet-B0 model
    """
    if device is None:
        device = get_device()
        
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                          'nvidia_efficientnet_b0', 
                          pretrained=pretrained, 
                          trust_repo=True)
    
    if train_mode:
        model.train()
    else:
        model.eval()
        
    return model.to(device)

def load_efficientnet_b4(pretrained=True, train_mode=True, device=None):
    """
    Load the EfficientNet-B4 model.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    device : torch.device
        Device to load the model on (if None, determined automatically)
        
    Returns:
    --------
    torch.nn.Module
        Loaded EfficientNet-B4 model
    """
    if device is None:
        device = get_device()
        
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                          'nvidia_efficientnet_b4', 
                          pretrained=pretrained, 
                          trust_repo=True)
    
    if train_mode:
        model.train()
    else:
        model.eval()
        
    return model.to(device)

def load_resnet_50(pretrained=True, train_mode=True, device=None):
    """
    Load the ResNet-50 model.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    device : torch.device
        Device to load the model on (if None, determined automatically)
        
    Returns:
    --------
    torch.nn.Module
        Loaded ResNet-50 model
    """
    if device is None:
        device = get_device()
        
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                          'nvidia_resnet50', 
                          pretrained=pretrained, 
                          trust_repo=True)
    
    if train_mode:
        model.train()
    else:
        model.eval()
        
    return model.to(device)

def load_resnext_101(pretrained=True, train_mode=True, device=None):
    """
    Load the ResNeXt-101 model.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    device : torch.device
        Device to load the model on (if None, determined automatically)
        
    Returns:
    --------
    torch.nn.Module
        Loaded ResNeXt-101 model
    """
    if device is None:
        device = get_device()
        
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                          'nvidia_resneXt', 
                          pretrained=pretrained, 
                          trust_repo=True)
    
    if train_mode:
        model.train()
    else:
        model.eval()
        
    return model.to(device)

def load_all_models(pretrained=True, train_mode=True, device=None):
    """
    Load all available models into a dictionary.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to load pretrained weights
    train_mode : bool
        Whether to set the model to training mode
    device : torch.device
        Device to load the models on (if None, determined automatically)
        
    Returns:
    --------
    dict
        Dictionary of loaded models with keys as model names
    """
    if device is None:
        device = get_device()
        
    models = {
        'efficientnet_b0': load_efficientnet_b0(pretrained, train_mode, device),
        'efficientnet_b4': load_efficientnet_b4(pretrained, train_mode, device),
        'resnet_50': load_resnet_50(pretrained, train_mode, device),
        'resnext_101': load_resnext_101(pretrained, train_mode, device)
    }
    
    return models

def print_model_summaries(models):
    """
    Print summary information for a dictionary of models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of PyTorch models
    """
    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        print(summary(model))