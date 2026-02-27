import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_directml
from preprocess_data import DatasetPreprocessor

device = torch_directml.device()

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 36),
            nn.LeakyReLU(),
            nn.Linear(36, 18),
            nn.LeakyReLU(),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.LeakyReLU(),
            nn.Linear(18, 36),
            nn.LeakyReLU(),
            nn.Linear(36, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(model, train_loader, test_loader, num_epochs, device, save_path, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device)
                loss = criterion(model(inputs), inputs)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | Val Loss: {avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping at epoch {epoch+1}. Best: {best_val_loss:.6f}")
                break

if __name__ == '__main__':
    datasets = [
        {"name": "UNSW", "path": "./src/datasets/unsw", "save": "best_autoencoder.pth"},
        {"name": "CIC", "path": "./src/datasets/cic", "save": "best_autoencoder_cic.pth"}
    ]

    for ds in datasets:
        print(f"\n--- Training Autoencoder on {ds['name']} ---")
        preprocessor = DatasetPreprocessor(ds['path'], ds['name'])
        
        train_t, test_t, labels_t, _ = preprocessor.get_tensors(save_as_pt=True)

        train_loader = DataLoader(TensorDataset(train_t), batch_size=4096, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_t), batch_size=4096, shuffle=False)

        model = Autoencoder(train_t.shape[1]).to(device)
        train_autoencoder(model, train_loader, test_loader, 150, device, ds['save'])