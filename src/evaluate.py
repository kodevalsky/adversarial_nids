import torch
from torch import nn
import torch_directml
import pandas as pd
from autoencoder import Autoencoder
from gan import Discriminator, Generator
from preprocess_data import DatasetPreprocessor
from attack import masked_fgsm_attack
from anogan import StaticAnoGAN, anogan_evaluate
from visualize import plot_feature_sensitivity, plot_unified_comparison, plot_inference_latency
from f_anogan import Encoder, F_AnoGAN
import time

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

def measure_inference_speed(ae, gen, disc, f_anogan_model, data_batch, device):
    """Measures the average time required to calculate the anomaly score for a single sample."""
    print("\n⏱️ Measuring Inference Latency...")
    
    test_batch = data_batch[:100].clone().detach().to(device)
    num_samples = len(test_batch)

    start_time = time.perf_counter()
    with torch.no_grad():
        _ = ae(test_batch)
    ae_time_per_sample = (time.perf_counter() - start_time) / num_samples

    start_time = time.perf_counter()
    with torch.no_grad():
        _ = f_anogan_model.get_anomaly_score(test_batch)
    fano_time_per_sample = (time.perf_counter() - start_time) / num_samples

    start_time = time.perf_counter()
    _, _ = anogan_evaluate(gen, disc, test_batch, device)
    ano_time_per_sample = (time.perf_counter() - start_time) / num_samples

    print("-" * 45)
    print(f"{'Model':<15} | {'Time per Sample (ms)':<20}")
    print("-" * 45)
    print(f"{'Autoencoder':<15} | {ae_time_per_sample * 1000:>10.4f} ms")
    print(f"{'Fast-AnoGAN':<15} | {fano_time_per_sample * 1000:>10.4f} ms")
    print(f"{'Standard AnoGAN':<15} | {ano_time_per_sample * 1000:>10.4f} ms")
    print("-" * 45)

    return ae_time_per_sample, fano_time_per_sample, ano_time_per_sample

def calculate_asr_from_scores(benign_scores, adv_scores):
    """Calculates Attack Success Rate strictly from pre-calculated score arrays."""
    threshold = torch.quantile(benign_scores, 0.95).item()
    successful_evasions = (adv_scores <= threshold).sum().item()
    asr = (successful_evasions / len(adv_scores)) * 100
    return asr, threshold

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

    benign_indices = (test_labels == 0).nonzero(as_tuple=True)[0]
    benign_batch = test_tensor[benign_indices][:1000].to(device)

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

    encoder = Encoder(input_dim).to(device)
    encoder.load_state_dict(torch.load(config['fano_encoder'], map_location=device))
    encoder.eval()

    f_anogan_model = F_AnoGAN(generator=gen, encoder=encoder, discriminator=disc).to(device)
    f_anogan_model.eval()

    # Autoencoder Testing
    with torch.no_grad():
        ae_base = criterion_mse(ae(attack_batch), attack_batch).mean().item()
    
    adv_ae = masked_fgsm_attack(attack_batch, ae, "AUTOENCODER", fgsm_mask, epsilon=0.05, device=device)
    
    with torch.no_grad():
        adv_ae_scores = criterion_mse(ae(adv_ae), adv_ae).mean(dim=1)
        ae_adv = adv_ae_scores.mean().item()
        benign_ae_scores = criterion_mse(ae(benign_batch), benign_batch).mean(dim=1)

    ae_asr, ae_threshold = calculate_asr_from_scores(benign_ae_scores, adv_ae_scores)

    # AnoGAN Testing
    base_anogan_scores, optimized_z = anogan_evaluate(gen, disc, attack_batch, device)
    anogan_base = base_anogan_scores.mean().item()

    anogan_target = StaticAnoGAN(gen, optimized_z).to(device)
    adv_anogan = masked_fgsm_attack(attack_batch, anogan_target, "AUTOENCODER", fgsm_mask, epsilon=0.05, device=device)
    
    adv_anogan_scores, _ = anogan_evaluate(gen, disc, adv_anogan, device)
    anogan_adv = adv_anogan_scores.mean().item()

    benign_ano_scores, _ = anogan_evaluate(gen, disc, benign_batch, device)

    anogan_asr, anogan_threshold = calculate_asr_from_scores(benign_ano_scores, adv_anogan_scores)
    
    # Fast-AnoGAN Testing
    with torch.no_grad():
        fano_base_scores = f_anogan_model(attack_batch)
        fano_base = fano_base_scores.mean().item()

    adv_f_anogan = masked_fgsm_attack(attack_batch, f_anogan_model, "F_ANOGAN", fgsm_mask, epsilon=0.05, device=device)

    with torch.no_grad():
        fano_adv_scores = f_anogan_model.get_anomaly_score(adv_f_anogan)
        fano_adv = fano_adv_scores.mean().item()
        benign_fa_scores = f_anogan_model.get_anomaly_score(benign_batch)
        
    fano_asr, fano_threshold = calculate_asr_from_scores(benign_fa_scores, fano_adv_scores)

    ae_time, fano_time, anogan_time = measure_inference_speed(ae, gen, disc, f_anogan_model, attack_batch, device)

    # Visualizations
    plot_feature_sensitivity(ae, attack_batch, feature_cols, criterion_mse, name)
    
    # Pass all 6 scores!
    plot_unified_comparison(ae_base, anogan_base, fano_base, ae_adv, anogan_adv, fano_adv, name)
    
    # Plot the latency!
    plot_inference_latency(ae_time, anogan_time, fano_time, name)

    return {
        "Dataset": name,
        "AE_Base": ae_base, "AE_Adv": ae_adv, "AE_Evasion": (1 - ae_adv/ae_base)*100, "AE_inf_time": ae_time, "AE_ASR": ae_asr, "AE_threshold": ae_threshold,
        "AG_Base": anogan_base, "AG_Adv": anogan_adv, "AG_Evasion": (1 - anogan_adv/anogan_base)*100, "AG_inf_time": anogan_time, "AG_ASR": anogan_asr, "AG_threshold": anogan_threshold,
        "FA_Base": fano_base, "FA_Adv": fano_adv, "FA_Evasion": (1 - fano_adv/fano_base)*100, "FA_inf_time": fano_time, "FA_ASR": fano_asr, "FA_threshold": fano_threshold,         
    }

def main():
    device = torch_directml.device()
    
    configs = [
        {
            'name': 'UNSW',
            'path': './src/datasets/unsw',
            'ae_path': './models/best_autoencoder_unsw.pth',
            'gen_path': './models/generator_final_unsw.pth',
            'disc_path': './models/discriminator_final_unsw.pth',
            'fano_encoder': './models/encoder_final_unsw.pth',
        },
        {
            'name': 'CIC',
            'path': './src/datasets/cic',
            'ae_path': './models/best_autoencoder_cic.pth',
            'gen_path': './models/generator_final_cic.pth',
            'disc_path': './models/discriminator_final_cic.pth',
            'fano_encoder': './models/encoder_final_cic.pth'
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
    print(f"\n{'#'*40}\n           FINAL ATTACK SUMMARY\n{'#'*40}")
    print(f"{'Dataset':<10} | {'Model':<12} | {'Base Score':<10} | {'Adv Score':<10} | {'Evasion %':<10} | {'ASR %':<8} | {'Inf Time'}")
    print("-" * 90)
    for s in summary_stats:
        print(f"{s['Dataset']:<10} | {'Autoencoder':<12} | {s['AE_Base']:<10.4f} | {s['AE_Adv']:<10.4f} | {s['AE_Evasion']:>8.2f}% | {s['AE_ASR']:>6.2f}% | {s['AE_inf_time']*1000:<8.4f} ms")
        print(f"{'':<10} | {'AnoGAN':<12} | {s['AG_Base']:<10.4f} | {s['AG_Adv']:<10.4f} | {s['AG_Evasion']:>8.2f}% | {s['AG_ASR']:>6.2f}% | {s['AG_inf_time']*1000:<8.4f} ms")
        print(f"{'':<10} | {'Fast-AnoGAN':<12} | {s['FA_Base']:<10.4f} | {s['FA_Adv']:<10.4f} | {s['FA_Evasion']:>8.2f}% | {s['FA_ASR']:>6.2f}% | {s['FA_inf_time']*1000:<8.4f} ms")
    print("-" * 90)
if __name__ == '__main__':
    main()