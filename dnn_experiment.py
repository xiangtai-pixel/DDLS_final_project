#!/usr/bin/env python
# coding: utf-8

# # Phase II (Advanced): FT-Transformer with Hyperparameter Search
# This script applies PCA, trains baselines, and runs a hyperparameter search for FT-Transformer.

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import rtdl
except ImportError:
    print("RTDL library not found. Please install it with: pip install rtdl")
    exit()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 2. Data Loading and Preparation ---
print('--- 1. Loading and Preparing Data ---')
try:
    proteomics_df = pd.read_csv('BRCA_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt', sep='\t', index_col=0)
    meta_df = pd.read_csv('BRCA_meta.txt', sep='\t', index_col=0)
    phenotype_df = pd.read_csv('BRCA_phenotype.txt', sep='\t', index_col=0)
except FileNotFoundError as e:
    print(f'Error: One or more data files not found: {e}')
    exit()

proteomics_df = proteomics_df.T
nan_ratio = proteomics_df.isna().mean()
proteomics_df = proteomics_df.loc[:, nan_ratio <= 0.3]
proteomics_df = proteomics_df.fillna(proteomics_df.mean())
df = pd.concat([meta_df, phenotype_df, proteomics_df], axis=1, join='inner')
df_clean = df.dropna(subset=['Stage'])

features = [col for col in proteomics_df.columns if col in df_clean.columns]
X = df_clean[features]
y = df_clean['Stage']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)

smallest_class_size = pd.Series(y_train_encoded).value_counts().min()
k = smallest_class_size - 1 if smallest_class_size > 1 else 1
smote = SMOTE(random_state=42, k_neighbors=k)
X_train_smote, y_train_smote_encoded = smote.fit_resample(X_train_scaled, y_train_encoded)

print('Data preparation complete.')

# --- 3. PCA for Dimensionality Reduction ---
print('\n--- 2. Applying PCA for Dimensionality Reduction ---')
n_components = 64
pca = PCA(n_components=n_components, random_state=42)

X_train_pca = pca.fit_transform(X_train_smote)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_baseline = PCA(n_components=n_components, random_state=42)
X_train_baseline_pca = pca_baseline.fit_transform(X_train_scaled)
X_test_baseline_pca = pca_baseline.transform(X_test_scaled)

print(f"PCA applied. Feature dimension reduced to {n_components}.")

# --- 4. Train Baseline Models on PCA Data ---
results = {}

print('\n--- 3. Training Baseline: Logistic Regression on PCA Data ---')
log_reg = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
log_reg.fit(X_train_baseline_pca, y_train)
y_pred_lr = log_reg.predict(X_test_baseline_pca)
lr_f1 = f1_score(y_test, y_pred_lr, average='macro', zero_division=0)
results['Logistic Regression + PCA'] = lr_f1
print(f"Logistic Regression + PCA Macro F1 on Test Set: {lr_f1:.4f}")

print('\n--- 4. Training Baseline: Random Forest with SMOTE on PCA Data ---')
y_train_smote = le.inverse_transform(y_train_smote_encoded)
rf_smote = RandomForestClassifier(random_state=42, n_estimators=100)
rf_smote.fit(X_train_pca, y_train_smote)
y_pred_rf = rf_smote.predict(X_test_pca)
rf_f1 = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
results['Random Forest + SMOTE + PCA'] = rf_f1
print(f"Random Forest + SMOTE + PCA Macro F1 on Test Set: {rf_f1:.4f}")

# --- 5. Hyperparameter Search for FT-Transformer ---
print('\n--- 5. Hyperparameter Search for FT-Transformer on PCA Data ---')

device = torch.device("cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_smote_encoded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Hyperparameter search space
learning_rates = [1e-5, 1e-4, 5e-4]
weight_decays = [1e-1, 1e-2]
best_hyperparams = {}
best_val_f1_overall = -1
best_model_state_overall = None

for lr in learning_rates:
    for wd in weight_decays:
        print(f"\n----- Testing combo: lr={lr}, wd={wd} ----- ")
        n_features_pca = n_components
        n_classes = len(le.classes_)

        ft_model = rtdl.FTTransformer.make_default(
            n_num_features=n_features_pca,
            cat_cardinalities=[],
            d_out=n_classes
        )
        ft_model.to(device)

        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        epochs = 150
        patience = 20
        patience_counter = 0
        best_val_f1_combo = -1
        best_model_state_combo = None

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
print(f"Best validation F1 score during search: {best_val_f1_overall:.4f}")
print(f"Best hyperparameters found: {best_hyperparams}")

# --- 6. Final Evaluation with Best Model ---
print("\n--- 6. Evaluating best model on Test Set ---")

final_model = rtdl.FTTransformer.make_default(
    n_num_features=n_features_pca,
    cat_cardinalities=[],
    d_out=n_classes
)
final_model.to(device)
final_model.load_state_dict(best_model_state_overall)

final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor.to(device), None)
    _, predicted_encoded = torch.max(test_outputs.data, 1)
    y_pred_final = le.inverse_transform(predicted_encoded.cpu().numpy())

ft_f1 = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
results['FT-Transformer + PCA (Tuned)'] = ft_f1
print(f"Tuned FT-Transformer + PCA Macro F1 on Test Set: {ft_f1:.4f}")
print("\n--- Tuned FT-Transformer + PCA Classification Report on Test Set ---")
print(classification_report(y_test, y_pred_final, zero_division=0))


# --- 7. Final Comparison ---
print('\n--- 7. Final Results Comparison (Macro F1 on Test Set) ---')
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
for model, score in sorted_results:
    print(f"{model}: {score:.4f}")