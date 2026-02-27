import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch

class DatasetPreprocessor:
    def __init__(self, data_path: str, dataset_type: str):
        self.data_path = data_path
        self.dataset_type = dataset_type.upper()
        self.feature_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
            'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 
            'swin', 'dwin', 'stcpb', 'dtcpb', 'smeans', 'dmeansz', 'trans_depth', 'res_bdy_len', 
            'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 
            'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]

    def _load_data(self) -> pd.DataFrame:
        """Loads all csv files and concatenates them into a single Dataframe."""
        datasets_container = []
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_path, file)
                if self.dataset_type == "UNSW" and "UNSW" in file.upper():
                    dataset = pd.read_csv(file_path, names=self.feature_names, low_memory=False)
                    datasets_container.append(dataset)
                elif self.dataset_type == "CIC" and "ISCX" in file.upper():
                    dataset = pd.read_csv(file_path)
                    datasets_container.append(dataset)
                    
        if datasets_container:
            return pd.concat(datasets_container, ignore_index=True)
        raise FileNotFoundError(f"No CSVs found for {self.dataset_type}")

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data by dropping invalid values, columns and converting non-categorical features to numeric."""
        df.columns = df.columns.str.strip()
        
        # Dropping irrelevant columns
        drops = {
            "UNSW": ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'attack_cat'],
            "CIC": ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp']
        }.get(self.dataset_type, [])
        df.drop(columns=drops, errors='ignore', inplace=True)
        
        # Handle infs and NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Convert all non-categorical columns to numeric
        for col in df.columns:
            if col not in ['proto', 'service', 'state', 'Label', 'label']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.fillna(0, inplace=True)
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes categorical features and collapses rare categories into 'Other'"""
        cat_map = {"UNSW": ['proto', 'service', 'state'], "CIC": []}
        categorical_cols = cat_map.get(self.dataset_type, [])
        
        valid_cols = [c for c in categorical_cols if c in df.columns]
        if not valid_cols: return df

        for col in valid_cols:
            df[col] = df[col].astype(str)
            top_5 = df[col].value_counts().index[:5]
            df[col] = df[col].apply(lambda x: x if x in top_5 else 'Other')
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_df = pd.DataFrame(encoder.fit_transform(df[valid_cols]), 
                                  columns=encoder.get_feature_names_out(valid_cols), 
                                  index=df.index)
        
        return pd.concat([df.drop(columns=valid_cols), encoded_df], axis=1)

    def _standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Makes sure the labels are consistent across datasets: 0 for benign, 1 for attack."""
        if "Label" in df.columns: 
            df.rename(columns={"Label": "label"}, inplace=True)
            df['label'] = df['label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        return df
    
    def get_tensors(self, save_as_pt: bool = False):
        """Returns the processed tensors along with the scaler for inverse transformations."""
        df = self._load_data()
        df = self._clean_dataset(df)
        df = self._encode_categoricals(df)
        df = self._standardize_labels(df)

        normal_data = df[df['label'] == 0]
        attack_data = df[df['label'] == 1]

        train_df = normal_data.sample(frac=0.8, random_state=42)
        test_unified = pd.concat([normal_data.drop(train_df.index), attack_data])

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df.drop(columns=['label']))
        test_scaled = scaler.transform(test_unified.drop(columns=['label']))

        tensors = (
            torch.tensor(train_scaled, dtype=torch.float32),
            torch.tensor(test_scaled, dtype=torch.float32),
            torch.tensor(test_unified['label'].values, dtype=torch.float32),
            scaler
        )

        if save_as_pt:
            torch.save(tensors[:3], f"{self.dataset_type.lower()}_tensors.pt")

        return tensors