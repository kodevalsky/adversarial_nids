import torch
from torch import nn

def masked_pgd_attack(data, model, model_type, mask, epsilon=0.05, alpha=0.01, num_iter=10, device="cpu"):
    random_noise = torch.empty_like(data).uniform_(-epsilon, epsilon).to(device)
    
    perturbed_data = data + (random_noise * mask)
    
    perturbed_data = torch.clamp(perturbed_data, 0.0, 1.0).detach()
    mask = mask.to(device)

    for i in range(num_iter):
        perturbed_data.requires_grad = True

        if model_type == "AUTOENCODER":
            outputs = model(perturbed_data)
            criterion = nn.MSELoss()
            loss = criterion(outputs, perturbed_data)
            
        elif model_type == "F_ANOGAN":
            loss = model.get_anomaly_score(perturbed_data).mean()
            
        elif model_type == "DISCRIMINATOR":
            outputs = model(perturbed_data)
            criterion = nn.BCELoss()
            target_labels = torch.ones_like(outputs).to(device)
            loss = criterion(outputs, target_labels)
            
        else:
            raise ValueError("Target model must be AUTOENCODER, DISCRIMINATOR, or F_ANOGAN")

        model.zero_grad()
        loss.backward()

        step = alpha * perturbed_data.grad.data.sign() * mask
        adv_data = perturbed_data - step

        eta = torch.clamp(adv_data - data, min=-epsilon, max=epsilon)
        
        perturbed_data = torch.clamp(data + eta, min=0.0, max=1.0).detach()

    return perturbed_data