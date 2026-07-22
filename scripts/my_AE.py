import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_dataset import MRRDataset
from src.my_processing import LogStandardScaler, MRRScaler
from src.model import Autoencoder
from src.my_plot import plot_mrr_comparison

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 1. ハイパーパラメータの設定 [2]
split_ratio = 0.8
epochs = 101
lr = 0.001
batch_size = 64
latent_dim = 2  # 潜在変数の次元数

# 波長の設定 (1545nm ~ 1555nm, 500ポイント)
wl = torch.linspace(1545e-9, 1555e-9, 500)

### 2. データセットの準備と分割 [4]
dataset = MRRDataset(
    num_samples=2000,
    tau1_range=[0.8, 0.99],
    tau2_range=[0.8, 0.99],
    alphas_range=[0.9, 0.99],
    FSR=0.01,
    wl=wl
)
train_raw, val_raw = random_split(dataset, [split_ratio, 1 - split_ratio])

# データの抽出
T_th_train, T_dr_train, params_train = train_raw[:]
T_th_val, T_dr_val, params_val = val_raw[:]

### 3. 前処理 (Scaling) [3, 5]
scaler_T = MRRScaler()
scaler_param = LogStandardScaler()

# スペクトル (Autoencoderの入出力) のスケーリング
# T_th と T_dr が結合され (N, 2 * len(wl)) の形状になる
T_train_scaled = scaler_T.fit(T_th_train, T_dr_train).transform(T_th_train, T_dr_train)
T_val_scaled = scaler_T.transform(T_th_val, T_dr_val)

# パラメータのスケーリング (可視化のラベル用)
params_val_scaled = scaler_param.fit_transform(params_val.numpy())

# DataLoaderの作成 [6]
train_loader = DataLoader(TensorDataset(T_train_scaled), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(T_val_scaled, torch.from_numpy(params_val_scaled).float()), 
                        batch_size=batch_size, shuffle=False)

### 4. モデル、損失関数、最適化手法の定義 [6, 7]
input_dim = T_train_scaled.shape[1] # 2 * len(wl)
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

### 5. 学習ループ [8, 9]
train_losses, val_losses = [], []
for epoch in range(epochs):
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
    
    # バリデーション
    model.eval()
    with torch.no_grad():
        v_loss = sum(criterion(model(x.to(device)), x.to(device)).item() for x, _ in val_loader) / len(val_loader)
        val_losses.append(v_loss)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")

### 6. 結果の可視化 [8, 10, 11]
# 学習曲線のプロット
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.title('Training History (Autoencoder)')
plt.legend()

# 復元精度の確認
model.eval()
T_actual_scaled, p_scaled = val_loader.dataset
T_pred_scaled = model(T_actual_scaled.unsqueeze(0).to(device)).squeeze(0)

# 逆変換してプロット [11, 12]
T_act_th, T_act_dr = scaler_T.inverse_transform(T_actual_scaled.unsqueeze(0))
T_pred_th, T_pred_dr = scaler_T.inverse_transform(T_pred_scaled.unsqueeze(0).detach())
p_act = scaler_param.inverse_transform(p_scaled.unsqueeze(0))

# T_th と T_dr を結合して比較関数に渡す
T_act_comb = torch.cat([T_act_th.flatten(), T_act_dr.flatten()])
T_pred_comb = torch.cat([T_pred_th.flatten(), T_pred_dr.flatten()])

plot_mrr_comparison(wl, T_act_comb, T_pred_comb, p_act)

### 潜在空間の可視化 [10]
T_all_val_scaled, _ = val_loader.dataset[:]
with torch.no_grad():
    z = model.encoder(T_all_val_scaled.to(device)).cpu().numpy()

plt.figure()
if latent_dim == 2:
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, c=params_val_scaled[:, 0], cmap='viridis')
    plt.colorbar(label='Scaled tau1')
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    plt.title('Latent Space (2D)')
elif latent_dim == 1:
    plt.hist(z, bins=50, alpha=0.7)
    plt.title('Latent Space Distribution (1D)')
plt.grid(True)
plt.show()