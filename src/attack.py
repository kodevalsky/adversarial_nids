import torch
from torch import nn

def masked_fgsm_attack(data, model, model_type, mask, epsilon=0.1, device="cpu"):
    """Executes a Targeted Masked FGSM attack to bypass anomaly detection."""
    data_copy = data.clone().detach().to(device)
    data_copy.requires_grad = True
    mask = mask.to(device)

    outputs = model(data_copy)

    if model_type == "AUTOENCODER":
        criterion = nn.MSELoss()
        loss = criterion(outputs, data_copy)
    elif model_type == "DISCRIMINATOR":
        criterion = nn.BCELoss()
        target_labels = torch.ones_like(outputs).to(device)
        loss = criterion(outputs, target_labels)
    else:
        raise ValueError("Target model must be AUTOENCODER or DISCRIMINATOR")

    model.zero_grad()
    loss.backward()

    masked_sign_grad = data_copy.grad.data.sign() * mask
    perturbed_data = data_copy - epsilon * masked_sign_grad

    return torch.clamp(perturbed_data, 0.0, 1.0).detach()