import torch
from torch import nn
import torch_directml
import pandas as pd
from autoencoder import Autoencoder
from gan import Discriminator, Generator
from preprocess_data import DatasetPreprocessor
from attack import masked_fgsm_attack
from anogan import StaticAnoGAN, anogan_evaluate
from visualize import plot_feature_sensitivity, plot_unified_comparison

def get_feature_columns(preprocessor):
    """Returns list of features for the dataset after preprocessing, excluding the label."""
    df_sample = preprocessor._load_data()
    df_sample = preprocessor._clean_dataset(df_sample)
    df_sample = preprocessor._encode_categoricals(df_sample)
    df_sample = preprocessor._standardize_labels(df_sample)
    return [c for c in df_sample.columns if c != 'label']

def create_mask(feature_cols, dataset_type):
    """Creates a mask to select features that can be safely mutated without breaking protocol semantics."""
    if dataset_type == "UNSW":
        mask = [0.0 if c.startswith(('proto_', 'service_', 'state_')) or 
                c in ['is_sm_ips_ports', 'is_ftp_login'] else 1.0 for c in feature_cols]
    else:
        mask = [0.0 if any(x in c for x in ['Flag', 'Binary', 'Id']) else 1.0 for c in feature_cols]
    return torch.tensor(mask, dtype=torch.float32)

def run_evaluation(config, device):
    """Main eval function"""
    name = config['name']
    print(f"\n{'='*20} EVALUATING DATASET: {name} {'='*20}")
    
    # Setup Data
    preprocessor = DatasetPreprocessor(config['path'], name)
    _, test_tensor, test_labels, scaler = preprocessor.get_tensors()
    input_dim = test_tensor.shape[1]
    feature_cols = get_feature_columns(preprocessor)
    fgsm_mask = create_mask(feature_cols, name).to(device)
    
    malicious_indices = (test_labels == 1).nonzero(as_tuple=True)[0]
    attack_batch = test_tensor[malicious_indices][:1000].to(device)
    criterion_mse = nn.MSELoss(reduction='none')

    # Load Models
    ae = Autoencoder(input_dim).to(device)
    ae.load_state_dict(torch.load(config['ae_path'], map_location=device))
    ae.eval()

    gen = Generator(64, input_dim).to(device)
    gen.load_state_dict(torch.load(config['gen_path'], map_location=device))
    gen.eval()

    disc = Discriminator(input_dim).to(device)
    disc.load_state_dict(torch.load(config['disc_path'], map_location=device))
    disc.eval()

    # Autoencoder Testing
    with torch.no_grad():
        ae_base = criterion_mse(ae(attack_batch), attack_batch).mean().item()
    
    adv_ae = masked_fgsm_attack(attack_batch, ae, "AUTOENCODER", fgsm_mask, epsilon=0.05, device=device)
    
    with torch.no_grad():
        ae_adv = criterion_mse(ae(adv_ae), adv_ae).mean().item()

    # AnoGAN Testing
    base_anogan_scores, optimized_z = anogan_evaluate(gen, disc, attack_batch, device)
    anogan_base = base_anogan_scores.mean().item()

    anogan_target = StaticAnoGAN(gen, optimized_z).to(device)
    adv_anogan = masked_fgsm_attack(attack_batch, anogan_target, "AUTOENCODER", fgsm_mask, epsilon=0.05, device=device)
    
    adv_anogan_scores, _ = anogan_evaluate(gen, disc, adv_anogan, device)
    anogan_adv = adv_anogan_scores.mean().item()

    # Visualizations
    plot_feature_sensitivity(ae, attack_batch, feature_cols, criterion_mse, name)
    plot_unified_comparison(ae_base, anogan_base, ae_adv, anogan_adv, name)

    return {
        "Dataset": name,
        "AE_Base": ae_base, "AE_Adv": ae_adv, "AE_Evasion": (1 - ae_adv/ae_base)*100,
        "AG_Base": anogan_base, "AG_Adv": anogan_adv, "AG_Evasion": (1 - anogan_adv/anogan_base)*100
    }

def main():
    device = torch_directml.device()
    
    configs = [
        {
            'name': 'UNSW',
            'path': './src/datasets/unsw',
            'ae_path': 'best_autoencoder.pth',
            'gen_path': 'generator_final.pth',
            'disc_path': 'discriminator_final.pth'
        },
        {
            'name': 'CIC',
            'path': './src/datasets/cic',
            'ae_path': 'best_autoencoder_cic.pth',
            'gen_path': 'generator_final_cic.pth',
            'disc_path': 'discriminator_final_cic.pth'
        }
    ]

    summary_stats = []
    for conf in configs:
        try:
            stats = run_evaluation(conf, device)
            summary_stats.append(stats)
        except FileNotFoundError as e:
            print(f"Skipping {conf['name']}: {e}")

    # Summary Table
    print(f"\n{'#'*30}\n      FINAL ATTACK SUMMARY\n{'#'*30}")
    print(f"{'Dataset':<10} | {'Model':<12} | {'Base Score':<10} | {'Adv Score':<10} | {'Evasion %'}")
    print("-" * 65)
    for s in summary_stats:
        print(f"{s['Dataset']:<10} | {'Autoencoder':<12} | {s['AE_Base']:<10.4f} | {s['AE_Adv']:<10.4f} | {s['AE_Evasion']:>8.2f}%")
        print(f"{'':<10} | {'AnoGAN':<12} | {s['AG_Base']:<10.4f} | {s['AG_Adv']:<10.4f} | {s['AG_Evasion']:>8.2f}%")
        print("-" * 65)

if __name__ == '__main__':
    main()