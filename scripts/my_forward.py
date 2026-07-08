import sys
from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_dataset import MRRDataset
from src.my_preprocessing import LogStandardScaler, MRRScaler
from src.model import MLP
from src.my_plot import plot_mrr_comparison

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 1. ハイパーパラメータの設定
split_ratio = 0.8
epochs = 51
lr = 0.001
batch_size = 64
hidden_size = 512

# 波長の設定 (1545nm ~ 1555nm, 1000ポイント)
wl = torch.linspace(1549.5e-9, 1550.5e-9, 1000)

### 2. データセットの準備と分割
dataset = MRRDataset(
    num_samples=1000,
    tau1_range=[0.8, 0.99],
    tau2_range=[0.8, 0.99],
    alpha_range=[0.9, 0.99],
    FSR=50e9,
    wl=wl
)
train_raw, val_raw = random_split(dataset, [split_ratio, 1 - split_ratio])

# 全データを抽出して前処理
T_th_train, T_dr_train, params_train = train_raw[:]
T_th_val, T_dr_val, params_val = val_raw[:]

### 3. 前処理 (Scaling)
scaler_param = LogStandardScaler()
scaler_T = MRRScaler()

# パラメータ (Input) のスケーリング
params_train_scaled = scaler_param.fit_transform(params_train.numpy())
params_val_scaled = scaler_param.transform(params_val.numpy())

# スペクトル (Target) のスケーリング
T_train_scaled = scaler_T.fit(T_th_train, T_dr_train).transform(T_th_train, T_dr_train)
T_val_scaled = scaler_T.transform(T_th_val, T_dr_val)

# DataLoaderの作成
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(params_train_scaled).float(), T_train_scaled),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(params_val_scaled).float(), T_val_scaled),
    batch_size=batch_size, shuffle=False
)

### 4. モデル、損失関数、最適化手法の定義
# 順問題: Params (4次元) -> Spectra (2 * len(wl)次元)
input_size = params_train.shape[1]
output_size = T_train_scaled.shape[1]

model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

### 5. 学習ループ
train_losses, val_losses = [], []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    train_losses.append(np.mean(batch_losses))
    
    # バリデーション
    model.eval()
    with torch.no_grad():
        v_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader) / len(val_loader)
        val_losses.append(v_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")

print(f"Training Time: {time.time() - start_time:.2f}s")

### 6. 結果の可視化
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.legend()
plt.title("Training History")

# 推論と逆変換の確認
# model.eval()
# idx = 0
# p_scaled, T_actual_scaled = val_loader.dataset[idx]
# T_pred_scaled = model(p_scaled.unsqueeze(0).to(device)).squeeze(0)

# # スケールを元に戻す
# T_actual_th, T_actual_dr = scaler_T.inverse_transform(T_actual_scaled.unsqueeze(0))
# T_pred_th, T_pred_dr = scaler_T.inverse_transform(T_pred_scaled.unsqueeze(0).detach())
# p_actual = scaler_param.inverse_transform(p_scaled.unsqueeze(0))

# # T_th と T_dr を結合して比較関数に渡す
# T_actual_comb = torch.cat([T_actual_th.flatten(), T_actual_dr.flatten()])
# T_pred_comb = torch.cat([T_pred_th.flatten(), T_pred_dr.flatten()])

# # # Tensor(修正前)
# # plot_mrr_comparison(wl, T_actual_comb, T_pred_comb, torch.from_numpy(p_actual))

# # Numpy(修正後)
# plot_mrr_comparison(wl, T_actual_comb, T_pred_comb, p_actual)
# # print(f"Actual Params: {p_actual}")
# # p_actual
# plt.show()

idx = [0, 1, 2]
p_actual_scaled, T_actual_scaled = val_loader.dataset[idx]