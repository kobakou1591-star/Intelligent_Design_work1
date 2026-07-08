import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# from src.my_dataset import RCDataset

class LogStandardScaler:
    """
    回路パラメータ (R, C, ...) に対する対数変換と Z-score 正規化を行うクラス。
    
    入力形式: (N, num_features) の Tensor. 
    仮定: 0 列目が R, 1 列目が C, ... (拡張可能)
    
    処理フロー:
    1. 入力：物理値 [R, C] (またはそれ以上の次元)
    2. 対数変換：log10(各パラメータ) を計算
    3. 正規化：StandardScaler で平均 0, 分散 1 に変換
    
    逆変換フロー:
    1. 入力：正規化されたデータ (N, num_features)
    2. 標準化の逆変換：log10 空間の値を復元
    3. 指数変換：10^x を計算して元の物理値 [R, C] を復元
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _validate_features(self, x: torch.Tensor) -> torch.Tensor:
        """形状の検証と整列 (N,) -> (N, 1), (N, num_features) 保証"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        if x.dim() == 1:
            x = x.unsqueeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Input must be 1D or 2D tensor, got {x.dim()}D")
            
        return x

    def fit(self, parameters: torch.Tensor):
        """
        訓練データから統計量を学習する。
        
        Args:
            parameters: 回路パラメータ (N, num_features). 
                        例: [R, C] の順に結合した tensor.
        """
        x = self._validate_features(parameters)
        
        # 対数変換 (各列ごとに独立に計算)
        log_x = torch.log10(x)
        
        # numpy 変換して sklearn に渡す
        inputs_log = log_x.numpy()
        
        self.scaler.fit(inputs_log)
        self.is_fitted = True
        return self

    def transform(self, parameters: torch.Tensor) -> np.ndarray:
        """
        対数変換と標準化を実行する。
        
        Args:
            parameters: (N, num_features)
            
        Returns:
            標準化されたデータ (N, num_features) numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("LogStandardScaler is not fitted. Call fit() first.")
            
        x = self._validate_features(parameters)
        log_x = torch.log10(x)
        
        inputs_log = log_x.numpy()
        
        return self.scaler.transform(inputs_log)

    def fit_transform(self, parameters: torch.Tensor) -> np.ndarray:
        """fit と transform をまとめて実行する。"""
        self.fit(parameters)
        return self.transform(parameters)

    def inverse_transform(self, scaled_data: np.ndarray) -> torch.Tensor:
        """
        標準化されたデータを元の物理値に戻す。
        
        Args:
            scaled_data: 標準化されたデータ (N, num_features)
            
        Returns:
            復元されたパラメータ (N, num_features) torch.Tensor
            [R, C, ...] の順で復元されます
        """
        if not self.is_fitted:
            raise RuntimeError("LogStandardScaler is not fitted. Call fit() first.")
            
        # 1. StandardScaler で log 空間の値を復元
        log_inputs = self.scaler.inverse_transform(scaled_data)
        
        # 2. 指数関数で元の物理値に戻す
        # numpy 配列のまま 10^x を計算し、Tensor に変換
        recovered_params = torch.tensor(10 ** log_inputs, dtype=torch.float32)
        
        return recovered_params
class MRRScaler:
    """
    MRRの透過スペクトル T_th, T_drをスケーリングするクラス。
    StandardScalarを用いて、全波長の平均と標準偏差を計算し、各波長での強度平均を0、標準偏差を1に変換する。
    T_th, T_drを水平方向に結合して、(N, 2 * num_wl)の形状に変換する。
    逆変換では、結合されたデータを分割し、標準化を逆変換して元の強度スケールに戻す。
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_wl = None

    def _combine_data(self, T_th, T_dr):
        """T_th と T_dr を水平方向に結合する内部メソッド"""
        # TensorをNumPyに変換
        if torch.is_tensor(T_th):
            T_th = T_th.detach().cpu().numpy()
        if torch.is_tensor(T_dr):
            T_dr = T_dr.detach().cpu().numpy()

        # 波長方向のサイズを保持 (inverse_transform用)
        if self.num_wl is None:
            self.num_wl = T_th.shape[2]
        
        # [T_th, T_dr] を横に結合 (shape: [samples, 2 * num_wl])
        return np.hstack([T_th, T_dr])

    def fit(self, T_th, T_dr):
        """学習データを用いて全波長の平均と標準偏差を計算"""
        combined = self._combine_data(T_th, T_dr)
        self.scaler.fit(combined)
        return self

    def transform(self, T_th, T_dr):
        """標準化を適用し、Tensorで返す"""
        combined = self._combine_data(T_th, T_dr)
        scaled = self.scaler.transform(combined)
        return torch.from_numpy(scaled).float()

    def inverse_transform(self, scaled_data):
        """標準化されたデータを元の強度スケールに戻す"""
        if torch.is_tensor(scaled_data):
            scaled_data = scaled_data.detach().cpu().numpy()
            
        inv_data = self.scaler.inverse_transform(scaled_data)
        
        # 結合したデータを半分に分けて戻す
        T_th = inv_data[:, :self.num_wl]
        T_dr = inv_data[:, self.num_wl:]
        
        return torch.from_numpy(T_th), torch.from_numpy(T_dr)

if __name__ == "__main__":
    from src.my_dataset import MRRDataset  # 自身で作成したデータセットをインポート
    
    print("--- MRRDataset 前処理の動作確認 ---")
    
    # 1. データセットの準備
    # 波長の設定 (1549.5nm ~ 1550.5nm)
    wl = torch.linspace(1549.5e-9, 1550.5e-9, 1000)
    dataset = MRRDataset(
        num_samples=1000,
        tau1_range=[0.8, 0.99],
        tau2_range=[0.8, 0.99],
        alphas_range=[0.9, 0.99],
        FSR=50e9,
        wl=wl
    )
    
    # 全データを取得
    # MRRDatasetの__getitem__が (T_th, T_dr, circuit_parameters) を返す想定 [1]
    T_th, T_dr, circuit_parameters = dataset[:] 
    
    # 2. 回路パラメータ (tau1, tau2, alpha, FSR) の前処理
    # LogStandardScaler を適用して対数変換と正規化を行う [2]
    param_scaler = LogStandardScaler()
    scaled_params = param_scaler.fit_transform(circuit_parameters.numpy())
    
    print(f"Original Params (first 2 samples):\n{circuit_parameters[:2]}")
    print(f"Scaled Params (first 2 samples):\n{scaled_params[:2]}")
    print(f"Scaled Params Shape: {scaled_params.shape}") # (1000, 4)
    
    # 3. スペクトルデータ (T_th, T_dr) の前処理
    # 以前作成した MRRScaler (実数版) を使用
    mrr_scaler = MRRScaler()
    mrr_scaler.fit(T_th, T_dr)
    scaled_spectra = mrr_scaler.transform(T_th, T_dr)
    
    print(f"\nScaled Spectra Shape: {scaled_spectra.shape}") # (1000, 2 * len(wl))
    
    # 4. 逆変換 (Inverse Transform) の確認
    # 推論結果を元の物理単位に戻せるか確認
    recovered_params = param_scaler.inverse_transform(scaled_params)
    recovered_T_th, recovered_T_dr = mrr_scaler.inverse_transform(scaled_spectra)
    
    # 誤差の確認 (オプション)
    param_error = np.abs(circuit_parameters.numpy() - recovered_params).max()
    print(f"\nMax reconstruction error (params): {param_error:.2e}")
    print("--- 動作確認完了 ---")