import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加 [5-7]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_dataset import MRRDataset
from src.my_preprocessing import LogStandardScaler, MRRScaler
from src.model import Autoencoder
from src.my_plot import plot_mrr_comparison

# デバイス設定 [6-9]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### 1. ハイパーパラメータの設定 [8, 10, 11]
split_ratio = 0.8
epochs = 501
lr = 0.001
batch_size = 128
latent_dim = 2

# 波長の設定 (1545nm ~ 1555nm, 500ポイント) [10, 11]
wl = torch.linspace(1549.5e-9, 1550.5e-9, 500)

##### 2. データセットの準備と分割 [10-12]
dataset = MRRDataset(
    num_samples=2000, 
    tau1_range=[0.8, 0.99], 
    tau2_range=[0.8, 0.99], 
    alpha_range=[0.9, 0.99], 
    FSR=50e9, 
    wl=wl
)
train_raw, val_raw = random_split(dataset, [split_ratio, 1 - split_ratio])

# データの抽出 [10-12]
T_th_train, T_dr_train, params_train = train_raw[:]
T_th_val, T_dr_val, params_val = val_raw[:]

##### 3. 前処理 (Scaling) [8, 9, 13-15]
scaler_T = MRRScaler()
scaler_param = LogStandardScaler()

# スペクトル (Autoencoderの入出力) のスケーリング [13-16]
T_train_scaled = scaler_T.fit(T_th_train, T_dr_train).transform(T_th_train, T_dr_train)
T_val_scaled = scaler_T.transform(T_th_val, T_dr_val)

# 潜在空間の色付け用にパラメータもスケーリング [8, 9, 13, 14]
params_val_scaled = scaler_param.fit_transform(params_val.numpy())

# DataLoaderの作成 (AEなので入力と教師は同じ T_train_scaled/T_val_scaled) [13-15, 17]
train_loader = DataLoader(TensorDataset(T_train_scaled), batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(T_val_scaled, torch.from_numpy(params_val_scaled).float())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

##### 4. モデル、損失関数、最適化手法の定義 [17-20]
input_dim = T_train_scaled.shape[1] # 2 * len(wl) [18, 19]
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

##### 5. 学習ループ (Lossのトラッキングを含む) [1, 4, 18, 19, 21]
train_losses, val_losses = [], []
for epoch in range(epochs):
    # Training
    model.train()
    batch_losses = []
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    # Validation
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.to(device)
            recon_val = model(x_val)
            loss_val = criterion(recon_val, x_val)
            val_batch_losses.append(loss_val.item())
    val_losses.append(np.mean(val_batch_losses))

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

##### 6. 結果の可視化

### (1) 学習曲線のプロット (AE.py や my_inv.py を参考) [1-4, 22]
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.yscale('log')
plt.title('Training History (Autoencoder)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

### (2) スペクトル比較プロット (my_inv.py のデータ構造を参考) [1-4, 22, 23]
model.eval()
idx = [0,1,2,3,4] # 比較したいサンプルのインデックス [1, 3, 4]
T_act_scaled, p_scaled = val_dataset[idx]

with torch.no_grad():
    T_pred_scaled = model(T_act_scaled.to(device))

# 逆変換
T_act_th, T_act_dr = scaler_T.inverse_transform(T_act_scaled.cpu())
T_pred_th, T_pred_dr = scaler_T.inverse_transform(T_pred_scaled.cpu())
p_act = scaler_param.inverse_transform(p_scaled.cpu())

# 1サンプルずつループしてプロット [3, 23]
for i in range(len(idx)):
    # Through と Drop を結合して (1000,) の形状にする [24-26]
    T_act_comb = torch.cat([torch.as_tensor(T_act_th[i]).flatten(), torch.as_tensor(T_act_dr[i]).flatten()])
    T_pred_comb = torch.cat([torch.as_tensor(T_pred_th[i]).flatten(), torch.as_tensor(T_pred_dr[i]).flatten()])
    
    # 2次元アクセス (params等) に対応するため (1, 3) 形状にする 
    p_sample = torch.as_tensor(p_act[i:i+1]).float()
    
    plot_mrr_comparison(wl, T_act_comb, T_pred_comb, p_sample)

# ### (3) 潜在空間の可視化 [24, 25, 27]
# T_all_val_scaled, _ = val_dataset[:]
# with torch.no_grad():
#     z = model.encoder(T_all_val_scaled.to(device)).cpu().numpy()

# plt.figure(figsize=(8, 6))
# if latent_dim == 2:
#     sc = plt.scatter(z[:, 0], z[:, 1], alpha=0.5, c=params_val_scaled[:, 0], cmap='viridis')
#     plt.colorbar(sc, label='Scaled tau1')
#     plt.xlabel('Latent Dimension 1')
#     plt.ylabel('Latent Dimension 2')
#     plt.title('Latent Space Distribution (MRR AE)')
# elif latent_dim == 1:
#     plt.hist(z.flatten(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
#     plt.title('Latent Space Distribution (1D Histogram)')

# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

##### 6. 潜在空間の可視化
T_all_val_scaled, _ = val_loader.dataset[:]
with torch.no_grad():
    z = model.encoder(T_all_val_scaled.to(device)).cpu().numpy()

fig = plt.figure(figsize=(10, 8))

if latent_dim == 3:
    # 3次元散布図の設定
    ax = fig.add_subplot(111, projection='3d')
    # パラメータの1つ目 (tau1) で色付け
    sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], alpha=0.6, c=params_val_scaled[:, 0], cmap='viridis')
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    ax.set_title('Latent Space Distribution (3D)')
    fig.colorbar(sc, ax=ax, label='Scaled tau1', pad=0.1)

elif latent_dim == 2:
    # 2次元散布図 (既存のソース[1, 2]を参考)
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, c=params_val_scaled[:, 0], cmap='viridis')
    plt.colorbar(label='Scaled tau1')
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    plt.title('Latent Space (2D)')

elif latent_dim == 1:
    # 1次元ヒストグラム (既存のソース[1, 2]を参考)
    plt.hist(z.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Latent Space Distribution (1D)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()