import matplotlib.pyplot as plt
import numpy as np

def print_extraction_table(orig_sample, adv_sample, scaler, feature_cols, mask_sum):
    """Inverse-transforms samples and prints physical feature deltas."""
    orig_real = scaler.inverse_transform(orig_sample.cpu().numpy().reshape(1, -1))[0]
    adv_real = scaler.inverse_transform(adv_sample.cpu().numpy().reshape(1, -1))[0]

    print(f"\n{'Feature Name':<20} | {'Original Value':<18} | {'Mutated Value':<18} | {'Delta':<10}")
    print("-" * 75)

    integer_cols = {'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'spkts', 'dpkts', 
                    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeans', 'dmeansz', 'trans_depth', 
                    'res_bdy_len', 'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd', 
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
                    'ct_dst_sport_ltm', 'ct_dst_src_ltm'}

    mutated_count = 0
    for i, col in enumerate(feature_cols):
        orig_val, adv_val = orig_real[i], adv_real[i]
        
        if col in integer_cols:
            orig_val, adv_val = round(orig_val), round(adv_val)
            
        delta = adv_val - orig_val
        if abs(delta) > 1e-4: 
            if col in integer_cols:
                print(f"{col:<20} | {int(orig_val):<18} | {int(adv_val):<18} | {int(delta):<10}")
            else:
                print(f"{col:<20} | {orig_val:<18.4f} | {adv_val:<18.4f} | {delta:<10.4f}")
            mutated_count += 1

    print("-" * 75)
    print(f"Total Features Mutated: {mutated_count} / {int(mask_sum)} allowed continuous features.")

def plot_feature_sensitivity(autoencoder, sample_batch, feature_cols, criterion_mse, dataset_name):
    """Plots top 20 gradients driving the Autoencoder's anomaly score with dataset-specific filename."""
    sample_to_explain = sample_batch[0:1].clone().detach().requires_grad_(True)
    
    autoencoder.zero_grad()
    loss_sample = criterion_mse(autoencoder(sample_to_explain), sample_to_explain).mean()
    loss_sample.backward()

    raw_gradients = sample_to_explain.grad[0].cpu().numpy()
    
    features, magnitudes = [], []
    for i, col in enumerate(feature_cols):
        if abs(raw_gradients[i]) > 1e-5:  
            features.append(col)
            magnitudes.append(raw_gradients[i])

    sorted_indices = np.argsort([abs(x) for x in magnitudes])[-20:]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_magnitudes = [magnitudes[i] for i in sorted_indices]
    colors = ['#ff4c4c' if x > 0 else '#3182bd' for x in sorted_magnitudes]

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_magnitudes, color=colors, edgecolor='black', height=0.6)
    plt.axvline(x=0, color='black', linewidth=1.5)
    
    plt.title(f'Feature Sensitivity Analysis ({dataset_name}): Autoencoder Gradients', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Gradient Magnitude (Impact on Anomaly Score)', fontsize=14, labelpad=10)
    plt.ylabel('Network Protocol Feature', fontsize=14, labelpad=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    filename = f'feature_sensitivity_{dataset_name.lower()}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"-> Saved '{filename}'")
    plt.close()

def plot_unified_comparison(ae_base, anogan_base, ae_adv, anogan_adv, dataset_name):
    """Plots baseline vs adversarial scores for both architectures with dataset-specific filename."""
    models = ['Deep Autoencoder', 'AnoGAN']
    baseline_scores = [ae_base, anogan_base]
    adv_scores = [ae_adv, anogan_adv]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Original Malware', color='#ff4c4c', edgecolor='black')
    rects2 = ax.bar(x + width/2, adv_scores, width, label='Adversarial Malware (Evaded)', color='#3182bd', edgecolor='black')

    ax.set_ylabel('Anomaly Score (Loss)', fontsize=14, labelpad=10)
    ax.set_title(f'Evasion Success ({dataset_name}): Autoencoder vs. AnoGAN', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    filename = f'unified_evasion_comparison_{dataset_name.lower()}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"-> Saved '{filename}'")
    plt.close()