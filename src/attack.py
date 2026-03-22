import torch
from torch import nn

def masked_fgsm_attack(data, model, model_type, mask, epsilon=0.1, device="cpu"):
    """Executes a Targeted Masked FGSM attack to bypass anomaly detection."""
    data_copy = data.clone().detach().to(device)
    data_copy.requires_grad = True
    mask = mask.to(device)

    if model_type == "AUTOENCODER":
        outputs = model(data_copy)
        criterion = nn.MSELoss()
        loss = criterion(outputs, data_copy)
        
    elif model_type == "F_ANOGAN":
        # Fast-AnoGAN calculates its own full anomaly score
        loss = model.get_anomaly_score(data_copy).mean()
        
    elif model_type == "DISCRIMINATOR":
        outputs = model(data_copy)
        criterion = nn.BCELoss()
        target_labels = torch.ones_like(outputs).to(device)
        loss = criterion(outputs, target_labels)
        
    else:
        raise ValueError("Target model must be AUTOENCODER, DISCRIMINATOR, or F_ANOGAN")

    model.zero_grad()
    loss.backward()

    masked_sign_grad = data_copy.grad.data.sign() * mask
    perturbed_data = data_copy - epsilon * masked_sign_grad

    return torch.clamp(perturbed_data, 0.0, 1.0).detach()