import torch
from torch import nn
import torch_directml
import pandas as pd
from autoencoder import Autoencoder
from gan import Discriminator, Generator
from preprocess_data import DatasetPreprocessor
from attack import masked_fgsm_attack
from pgd import masked_pgd_attack
from anogan import StaticAnoGAN, anogan_evaluate
from visualize import plot_feature_sensitivity, plot_unified_comparison, plot_inference_latency
from f_anogan import Encoder, F_AnoGAN
from utils import set_deterministic_seed
import statistics
import time

def get_feature_columns(preprocessor):
    df_sample = preprocessor._load_data()
    df_sample = preprocessor._clean_dataset(df_sample)
    df_sample = preprocessor._encode_categoricals(df_sample)
    df_sample = preprocessor._standardize_labels(df_sample)
    return [c for c in df_sample.columns if c != 'label']

def create_mask(feature_cols, dataset_type):
    if dataset_type == "UNSW":
        mask = [0.0 if c.startswith(('proto_', 'service_', 'state_')) or 
                c in ['is_sm_ips_ports', 'is_ftp_login'] else 1.0 for c in feature_cols]
    else:
        mask = [0.0 if any(x in c for x in ['Flag', 'Binary', 'Id']) else 1.0 for c in feature_cols]
    return torch.tensor(mask, dtype=torch.float32)

def measure_inference_speed(ae, gen, disc, f_anogan_model, data_batch, device):
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
    threshold = torch.quantile(benign_scores, 0.95).item()
    successful_evasions = (adv_scores <= threshold).sum().item()
    asr = (successful_evasions / len(adv_scores)) * 100
    return asr, threshold

def run_evaluation(config, device, seed, is_first_seed):
    name = config['name']
    print(f"\n{'='*20} EVALUATING DATASET: {name} {'='*20}")
    
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

    # Define PGD hyperparameters
    eps = 0.05
    alpha = 0.0125
    iters = 10

    # --- Autoencoder Testing ---
    with torch.no_grad():
        ae_base = criterion_mse(ae(attack_batch), attack_batch).mean().item()
        benign_ae_scores = criterion_mse(ae(benign_batch), benign_batch).mean(dim=1)
    
    adv_ae_fgsm = masked_fgsm_attack(attack_batch, ae, "AUTOENCODER", fgsm_mask, epsilon=eps, device=device)
    adv_ae_pgd = masked_pgd_attack(attack_batch, ae, "AUTOENCODER", fgsm_mask, epsilon=eps, alpha=alpha, num_iter=iters, device=device)
    
    with torch.no_grad():
        adv_ae_scores_fgsm = criterion_mse(ae(adv_ae_fgsm), adv_ae_fgsm).mean(dim=1)
        ae_adv_fgsm = adv_ae_scores_fgsm.mean().item()
        
        adv_ae_scores_pgd = criterion_mse(ae(adv_ae_pgd), adv_ae_pgd).mean(dim=1)
        ae_adv_pgd = adv_ae_scores_pgd.mean().item()

    ae_asr_fgsm, _ = calculate_asr_from_scores(benign_ae_scores, adv_ae_scores_fgsm)
    ae_asr_pgd, _ = calculate_asr_from_scores(benign_ae_scores, adv_ae_scores_pgd)

    # --- AnoGAN Testing ---
    base_anogan_scores, optimized_z = anogan_evaluate(gen, disc, attack_batch, device)
    anogan_base = base_anogan_scores.mean().item()
    benign_ano_scores, _ = anogan_evaluate(gen, disc, benign_batch, device)

    anogan_target = StaticAnoGAN(gen, optimized_z).to(device)
    adv_anogan_fgsm = masked_fgsm_attack(attack_batch, anogan_target, "AUTOENCODER", fgsm_mask, epsilon=eps, device=device)
    adv_anogan_pgd = masked_pgd_attack(attack_batch, anogan_target, "AUTOENCODER", fgsm_mask, epsilon=eps, alpha=alpha, num_iter=iters, device=device)
    
    adv_anogan_scores_fgsm, _ = anogan_evaluate(gen, disc, adv_anogan_fgsm, device)
    anogan_adv_fgsm = adv_anogan_scores_fgsm.mean().item()
    
    adv_anogan_scores_pgd, _ = anogan_evaluate(gen, disc, adv_anogan_pgd, device)
    anogan_adv_pgd = adv_anogan_scores_pgd.mean().item()

    anogan_asr_fgsm, _ = calculate_asr_from_scores(benign_ano_scores, adv_anogan_scores_fgsm)
    anogan_asr_pgd, _ = calculate_asr_from_scores(benign_ano_scores, adv_anogan_scores_pgd)
    
    # --- Fast-AnoGAN Testing ---
    with torch.no_grad():
        fano_base_scores = f_anogan_model(attack_batch)
        fano_base = fano_base_scores.mean().item()
        benign_fa_scores = f_anogan_model.get_anomaly_score(benign_batch)

    adv_fano_fgsm = masked_fgsm_attack(attack_batch, f_anogan_model, "F_ANOGAN", fgsm_mask, epsilon=eps, device=device)
    adv_fano_pgd = masked_pgd_attack(attack_batch, f_anogan_model, "F_ANOGAN", fgsm_mask, epsilon=eps, alpha=alpha, num_iter=iters, device=device)

    with torch.no_grad():
        fano_adv_scores_fgsm = f_anogan_model.get_anomaly_score(adv_fano_fgsm)
        fano_adv_fgsm = fano_adv_scores_fgsm.mean().item()
        
        fano_adv_scores_pgd = f_anogan_model.get_anomaly_score(adv_fano_pgd)
        fano_adv_pgd = fano_adv_scores_pgd.mean().item()
        
    fano_asr_fgsm, _ = calculate_asr_from_scores(benign_fa_scores, fano_adv_scores_fgsm)
    fano_asr_pgd, _ = calculate_asr_from_scores(benign_fa_scores, fano_adv_scores_pgd)

    ae_time, fano_time, anogan_time = measure_inference_speed(ae, gen, disc, f_anogan_model, attack_batch, device)

    # Visualizations (Only trigger on the first seed)
    if is_first_seed:
        plot_feature_sensitivity(ae, attack_batch, feature_cols, criterion_mse, name)
        # Updated to pass both FGSM and PGD scores
        plot_unified_comparison(ae_base, anogan_base, fano_base, ae_adv_fgsm, anogan_adv_fgsm, fano_adv_fgsm, ae_adv_pgd, anogan_adv_pgd, fano_adv_pgd, name)
        plot_inference_latency(ae_time, anogan_time, fano_time, name)

    return {
        "Dataset": name,
        "AE_inf_time": ae_time, "AG_inf_time": anogan_time, "FA_inf_time": fano_time,
        
        "AE_FGSM_Ev": (1 - ae_adv_fgsm/ae_base)*100, "AE_FGSM_ASR": ae_asr_fgsm,
        "AE_PGD_Ev": (1 - ae_adv_pgd/ae_base)*100,   "AE_PGD_ASR": ae_asr_pgd,
        
        "AG_FGSM_Ev": (1 - anogan_adv_fgsm/anogan_base)*100, "AG_FGSM_ASR": anogan_asr_fgsm,
        "AG_PGD_Ev": (1 - anogan_adv_pgd/anogan_base)*100,   "AG_PGD_ASR": anogan_asr_pgd,
        
        "FA_FGSM_Ev": (1 - fano_adv_fgsm/fano_base)*100, "FA_FGSM_ASR": fano_asr_fgsm,
        "FA_PGD_Ev": (1 - fano_adv_pgd/fano_base)*100,   "FA_PGD_ASR": fano_asr_pgd, 
    }

def main():
    device = torch_directml.device()
    SEEDS = [42, 123, 2026]
    dataset_names = ['UNSW', 'CIC']
    
    aggregated_results = {name: [] for name in dataset_names}

    for i, seed in enumerate(SEEDS):
        set_deterministic_seed(seed)
        is_first_seed = (i == 0)
        
        for name in dataset_names:
            config = {
                'name': name,
                'path': f'./src/datasets/{name.lower()}',
                'ae_path': f'./models/best_autoencoder_{name.lower()}_seed{seed}.pth',
                'gen_path': f'./models/generator_final_{name.lower()}_seed{seed}.pth',
                'disc_path': f'./models/discriminator_final_{name.lower()}_seed{seed}.pth',
                'fano_encoder': f'./models/encoder_final_{name.lower()}_seed{seed}.pth'
            }
            try:
                stats = run_evaluation(config, device, seed, is_first_seed)
                aggregated_results[name].append(stats)
            except FileNotFoundError as e:
                print(f"Skipping {name} for seed {seed}: {e}")

    print(f"\n{'#'*95}\n                                   FGSM vs PGD SUMMARY\n{'#'*95}")
    print(f"{'Dataset':<8} | {'Model':<12} | {'FGSM Evas %':<14} | {'PGD Evas %':<14} | {'FGSM ASR':<10} | {'PGD ASR':<10} | {'Inf Time'}")
    print("-" * 95)
    
    for name in dataset_names:
        if not aggregated_results[name]: continue
        runs = aggregated_results[name]
        
        for prefix, model_name in [('AE', 'Autoencoder'), ('AG', 'AnoGAN'), ('FA', 'Fast-AnoGAN')]:
            m_f_ev = statistics.mean([r[f'{prefix}_FGSM_Ev'] for r in runs])
            s_f_ev = statistics.stdev([r[f'{prefix}_FGSM_Ev'] for r in runs])
            
            m_p_ev = statistics.mean([r[f'{prefix}_PGD_Ev'] for r in runs])
            s_p_ev = statistics.stdev([r[f'{prefix}_PGD_Ev'] for r in runs])
            
            m_f_asr = statistics.mean([r[f'{prefix}_FGSM_ASR'] for r in runs])
            s_f_asr = statistics.stdev([r[f'{prefix}_FGSM_ASR'] for r in runs]) if m_f_asr > 0 else 0.0
            
            m_p_asr = statistics.mean([r[f'{prefix}_PGD_ASR'] for r in runs])
            s_p_asr = statistics.stdev([r[f'{prefix}_PGD_ASR'] for r in runs]) if m_p_asr > 0 else 0.0
            
            m_inf = statistics.mean([r[f'{prefix}_inf_time'] * 1000 for r in runs])
            s_inf = statistics.stdev([r[f'{prefix}_inf_time'] * 1000 for r in runs])
            
            print(f"{name:<8} | {model_name:<12} | {m_f_ev:>5.1f}±{s_f_ev:<4.1f}% | {m_p_ev:>5.1f}±{s_p_ev:<4.1f}% | {m_f_asr:>4.1f}±{s_f_asr:<3.1f}% | {m_p_asr:>4.1f}±{s_p_asr:<3.1f}% | {m_inf:>6.4f}")
    print("-" * 95)

if __name__ == '__main__':
    main()