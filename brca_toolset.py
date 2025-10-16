#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

try:
    import rtdl
except ImportError:
    print("RTDL library not found. Please install it with: pip install rtdl")
    exit()

# MCP framework (assuming a simple argparse-based implementation if FastMCP is not available)
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Define a fallback CLI handler if FastMCP is not installed
    import argparse

    class FallbackMCP:
        def __init__(self, description):
            self.parser = argparse.ArgumentParser(description=description)
            self.subparsers = self.parser.add_subparsers(dest="command", required=True)
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                # Basic argument parsing from type hints
                # This is a simplified version and may not handle all cases
                subparser = self.subparsers.add_parser(func.__name__, help=func.__doc__)
                return func
            return decorator

        def run(self, transport=None):
            args = self.parser.parse_args()
            if args.command in self.tools:
                # Simplified: does not handle tool arguments
                self.tools[args.command]()
            else:
                self.parser.print_help()

    FastMCP = FallbackMCP


# --- Configuration ---
warnings.filterwarnings('ignore')
mcp = FastMCP("BRCA Proteomics FT-Transformer Pipeline")

ARTIFACTS_DIR = 'brca_pipeline_artifacts'

# --- Tool Definitions ---

@mcp.tool()
def preprocess():
    """Loads raw data, preprocesses, applies SMOTE+PCA, and saves all artifacts."""
    print('--- 1. Starting Data Preprocessing ---')
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load data
    try:
        proteomics_df = pd.read_csv('BRCA_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt', sep='\t', index_col=0)
        meta_df = pd.read_csv('BRCA_meta.txt', sep='\t', index_col=0)
        phenotype_df = pd.read_csv('BRCA_phenotype.txt', sep='\t', index_col=0)
    except FileNotFoundError as e:
        print(f'Error: One or more data files not found: {e}')
        return

    # Basic cleaning
    proteomics_df = proteomics_df.T
    nan_ratio = proteomics_df.isna().mean()
    proteomics_df = proteomics_df.loc[:, nan_ratio <= 0.3]
    proteomics_df = proteomics_df.fillna(proteomics_df.mean())
    df = pd.concat([meta_df, phenotype_df, proteomics_df], axis=1, join='inner')
    df_clean = df.dropna(subset=['Stage'])

    features_cols = [col for col in proteomics_df.columns if col in df_clean.columns]
    X = df_clean[features_cols]
    y = df_clean['Stage']

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Label Encoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # SMOTE
    smallest_class_size = pd.Series(y_train_encoded).value_counts().min()
    k = smallest_class_size - 1 if smallest_class_size > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_smote, y_train_smote_encoded = smote.fit_resample(X_train_scaled, y_train_encoded)

    # PCA
    n_components = 128
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_smote)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Save all artifacts
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(ARTIFACTS_DIR, 'pca.joblib'))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib'))
    np.save(os.path.join(ARTIFACTS_DIR, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_train_smote_encoded.npy'), y_train_smote_encoded)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_val_pca.npy'), X_val_pca)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_val_encoded.npy'), y_val_encoded)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_test_pca.npy'), X_test_pca)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_test_encoded.npy'), y_test_encoded)

    print(f"Preprocessing complete. All artifacts saved to '{ARTIFACTS_DIR}' directory.")

@mcp.tool()
def train():
    """Runs hyperparameter search for the FT-Transformer and saves the best model."""
    print('\n--- 2. Starting Model Training & Hyperparameter Search ---')
    
    # Load preprocessed data
    try:
        X_train_pca = np.load(os.path.join(ARTIFACTS_DIR, 'X_train_pca.npy'))
        y_train_smote_encoded = np.load(os.path.join(ARTIFACTS_DIR, 'y_train_smote_encoded.npy'))
        X_val_pca = np.load(os.path.join(ARTIFACTS_DIR, 'X_val_pca.npy'))
        y_val_encoded = np.load(os.path.join(ARTIFACTS_DIR, 'y_val_encoded.npy'))
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run the 'preprocess' command first.")
        return

    device = torch.device("cpu")
    print(f"Using device: {device}")

    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_smote_encoded, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Hyperparameter search
    learning_rates = [1e-5, 1e-4, 5e-4]
    weight_decays = [1e-1, 1e-2]
    best_hyperparams = {}
    best_val_f1_overall = -1
    best_model_state_overall = None

    for lr in learning_rates:
        for wd in weight_decays:
            print(f"\n----- Testing combo: lr={lr}, wd={wd} ----- ")
            n_features_pca = X_train_pca.shape[1]
            n_classes = len(np.unique(y_train_smote_encoded))

            ft_model = rtdl.FTTransformer.make_default(n_num_features=n_features_pca, cat_cardinalities=[], d_out=n_classes)
            ft_model.to(device)

            optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()

            epochs, patience, patience_counter = 150, 20, 0
            best_val_f1_combo, best_model_state_combo = -1, None

            for epoch in range(epochs):
                ft_model.train()
                for x_cont, y_batch in train_loader:
                    x_cont, y_batch = x_cont.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = ft_model(x_cont, None)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                ft_model.eval()
                with torch.no_grad():
                    val_outputs = ft_model(X_val_tensor.to(device), None)
                    _, predicted = torch.max(val_outputs.data, 1)
                    val_f1 = f1_score(y_val_tensor.numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

                if val_f1 > best_val_f1_combo:
                    best_val_f1_combo = val_f1
                    patience_counter = 0
                    best_model_state_combo = ft_model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            print(f"Result for combo: Best Val F1 = {best_val_f1_combo:.4f}")
            if best_val_f1_combo > best_val_f1_overall:
                best_val_f1_overall = best_val_f1_combo
                best_hyperparams = {'lr': lr, 'wd': wd}
                best_model_state_overall = best_model_state_combo

    print(f"\n--- Hyperparameter Search Finished ---")
    print(f"Best validation F1: {best_val_f1_overall:.4f}")
    print(f"Best hyperparameters: {best_hyperparams}")

    # Save the best model and its params
    torch.save(best_model_state_overall, os.path.join(ARTIFACTS_DIR, 'best_ft_transformer.pt'))
    with open(os.path.join(ARTIFACTS_DIR, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_hyperparams, f)
    
    print("Best model and hyperparameters saved.")

@mcp.tool()
def predict():
    """Loads the best model and runs inference on the test set."""
    print('\n--- 3. Making Predictions on Test Set ---')
    try:
        X_test_pca = np.load(os.path.join(ARTIFACTS_DIR, 'X_test_pca.npy'))
        le = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib'))
        model_state = torch.load(os.path.join(ARTIFACTS_DIR, 'best_ft_transformer.pt'))
    except FileNotFoundError:
        print("Error: Necessary artifacts not found. Please run 'preprocess' and 'train' first.")
        return

    n_features_pca = X_test_pca.shape[1]
    n_classes = len(le.classes_)

    model = rtdl.FTTransformer.make_default(n_num_features=n_features_pca, cat_cardinalities=[], d_out=n_classes)
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test_pca, dtype=torch.float32), None)
        _, predicted_encoded = torch.max(test_outputs.data, 1)
        predictions = le.inverse_transform(predicted_encoded.cpu().numpy())

    np.save(os.path.join(ARTIFACTS_DIR, 'predictions.npy'), predictions)
    print(f"Predictions saved to '{os.path.join(ARTIFACTS_DIR, 'predictions.npy')}'")

@mcp.tool()
def evaluate():
    """Evaluates the saved predictions against the true test labels."""
    print('\n--- 4. Evaluating Model Performance ---')
    try:
        predictions = np.load(os.path.join(ARTIFACTS_DIR, 'predictions.npy'))
        y_test_encoded = np.load(os.path.join(ARTIFACTS_DIR, 'y_test_encoded.npy'))
        le = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib'))
    except FileNotFoundError:
        print("Error: Necessary artifacts not found. Please run 'preprocess', 'train', and 'predict' first.")
        return

    y_test = le.inverse_transform(y_test_encoded)
    print("--- Final Model Report on Test Set ---")
    print(classification_report(y_test, predictions, zero_division=0))
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    print(f"Final Macro F1 on Test Set: {f1:.4f}")

if __name__ == "__main__":
    # This allows the script to be run directly with command-line arguments
    # e.g., python brca_toolset.py preprocess
    mcp.run(transport="stdio")
