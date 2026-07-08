import matplotlib.pyplot as plt
import numpy as np
import torch
from src.my_dataset import MRRDataset

def plot_mrr_response(mrr_dataset: MRRDataset, num_samples: int | list):
    """
    MRRDataset内のサンプルから透過スペクトル (T_th, T_dr) をプロットする。
    """
    # 波長をnm単位に変換して取得
    wl_nm = mrr_dataset.wl.numpy() * 1e9
    
    # 指定された数またはリストに基づきインデックスを取得
    if isinstance(num_samples, int):
        indices = np.random.choice(len(mrr_dataset), num_samples, replace=False)
    else:
        indices = num_samples

    # データの取得 (T_th, T_dr, params) [2, 3]
    T_th_all, T_dr_all, params_all = mrr_dataset[indices]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i in range(len(indices)):
        p = params_all[i]
        # ラベルにパラメータを表示 (tau1, tau2, alpha)
        label = f"t1:{p[0]:.2f}, t2:{p[1]:.2f}, a:{p[2]:.2f}"
        
        axes.plot(wl_nm, T_th_all[i], label=label)
        axes[1].plot(wl_nm, T_dr_all[i], label=label)

    # Throughポートのプロット設定
    axes.set_ylabel("Through Transmission ($T_{th}$)")
    axes.set_title("MRR Spectral Response")
    axes.grid(True)
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Dropポートのプロット設定
    axes[1].set_ylabel("Drop Transmission ($T_{dr}$)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_mrr_comparison(wl: torch.Tensor, T_actual: torch.Tensor, T_pred: torch.Tensor, params: torch.Tensor):
    """
    実際のスペクトルと予測されたスペクトルを比較プロットする。
    T_actual/T_pred は [T_th, T_dr] が結合された形式 (2 * num_wl) を想定 [3]。
    """
    wl_np = wl.detach().cpu().numpy() * 1e9
    num_wl = len(wl_np)
    
    # 結合されたデータを Through と Drop に分離
    T_th_act = T_actual[:num_wl].detach().cpu().numpy()
    T_dr_act = T_actual[num_wl:].detach().cpu().numpy()
    T_th_pred = T_pred[:num_wl].detach().cpu().numpy()
    T_dr_pred = T_pred[num_wl:].detach().cpu().numpy()
    
    tau1=params[0][0]
    tau2=params[0][1]
    alpha=params[0][2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Through 比較
    axes[0].plot(wl_np, T_th_act, 'k-', label="Actual")
    axes[0].plot(wl_np, T_th_pred, 'r--', label="Predicted")
    axes[0].set_ylabel("$T_{th}$")
    axes[0].set_title(f"Comparison - t1:{tau1:.3f}, t2:{tau2:.3f}, a:{alpha:.3f}")
    axes[0].legend()
    axes[0].grid(True)

    # Drop 比較
    axes[1].plot(wl_np, T_dr_act, 'k-', label="Actual")
    axes[1].plot(wl_np, T_dr_pred, 'b--', label="Predicted")
    axes[1].set_ylabel("$T_{dr}$")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 動作確認用
    wl = torch.linspace(1545e-9, 1555e-9, 100)
    dataset = MRRDataset(
        num_samples=10,
        tau1_range=[0.8, 0.99],
        tau2_range=[0.8, 0.99],
        alphas_range=[0.9, 0.99],
        FSR=0.01,
        wl=wl
    )
    
    # データセットの可視化
    plot_mrr_response(dataset, num_samples=3)