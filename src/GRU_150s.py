import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# === Dataset class ===
class ThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name
        columns_for_scaling = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        rename_map = {
            "T_ave (C)": "T_avg (C)",
        }
        df.rename(columns=rename_map, inplace=True)
        if "Input Temperature (C)" not in df.columns and "T_inner (C)" in df.columns:
            df["Input Temperature (C)"] = df["T_inner (C)"]
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler
        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])
        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.full_time, self.full_t_min, self.full_t_max, self.full_t_ave = [], [], [], []

        for _, group in grouped:
            # Input: Time, T_outer, Input Temperature, T_avg (replace T_inner with Input Temperature)
            X_seq = group[["Time (s)", "T_outer (C)", "Input Temperature (C)", "T_avg (C)"]].values[:-1]
            # Output: T_inner, T_outer, T_avg (3 temperatures)
            Y_seq = group[["T_inner (C)", "T_outer (C)", "T_avg (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]
            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_min.append(group["T_outer (C)"].values)
            self.full_t_max.append(group["T_inner (C)"].values)
            self.full_t_ave.append(group["T_avg (C)"].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.full_time = np.array(self.full_time)
        self.full_t_min = np.array(self.full_t_min)
        self.full_t_max = np.array(self.full_t_max)
        self.full_t_ave = np.array(self.full_t_ave)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx], self.Y[idx], self.time_values[idx],
            self.full_time[idx],
            self.full_t_min[idx], self.full_t_max[idx], self.full_t_ave[idx]
        )


# === Model definition ===
class ThermalGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=3, num_layers=5):
        super(ThermalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# === Weighted loss ===
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0, 1.0]), time_weights=None):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * weights
    if time_weights is not None:
        time_weights = time_weights.to(predictions.device)
        loss = loss * time_weights.unsqueeze(-1)
    return torch.mean(loss)


# === Training function ===
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    possible_data_dirs = [
        os.path.normpath(os.path.join(script_dir, "..", "data")),   # 根/data
        os.path.join(script_dir, "data"),                           # src/data
        os.path.abspath(os.path.join(os.getcwd(), "data")),         # 工作目录/data
    ]

    data_dir = next((p for p in possible_data_dirs if os.path.isdir(p)), None)
    print("script_dir =", script_dir)
    print("checked data roots =", possible_data_dirs)
    print("selected data_dir =", data_dir)

    if not data_dir:
        print("Data directory not found")
        return None, None

    # 兼容不同目录名
    train_candidates = [
        os.path.join(data_dir, "150s Time steps"),
        os.path.join(data_dir, "data_in_150s"),
    ]
    test_candidates = [
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "test_in_150s"),
    ]
    train_dir = next((p for p in train_candidates if os.path.isdir(p)), None)
    test_dir  = next((p for p in test_candidates  if os.path.isdir(p)), None)

    print("selected train_dir =", train_dir)
    print("selected test_dir  =", test_dir)

    # 在选出 train_dir / test_dir 之后，立刻枚举文件
    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths  = sorted(glob.glob(os.path.join(test_dir, "*.csv"))) if test_dir else []

    print(f"Found training files: {len(train_paths)}")
    print(f"Found test files: {len(test_paths)}")

    if not train_paths:
        print("No training files found")
        return None, None

    if not train_dir:
        print("No training dir. Tried:", train_candidates)
        return None, None
    if not test_dir:
        print("No test dir. Tried:", test_candidates)
        test_dir = None

    # === Split out validation set (optional 5%) ===
    val_split = max(1, int(0.05 * len(train_paths))) if len(train_paths) > 1 else 0
    val_paths = train_paths[:val_split] if val_split > 0 else []
    actual_train_paths = train_paths[val_split:] if val_split > 0 else train_paths

    # === Fit a unified scaler on training data ===
    train_dfs = [pd.read_csv(f) for f in actual_train_paths]
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_dfs)[["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)"]])

    train_datasets = [ThermalDataset(f, scaler=scaler) for f in actual_train_paths]
    val_datasets = [ThermalDataset(f, scaler=scaler) for f in val_paths]
    test_datasets = [ThermalDataset(f, scaler=scaler) for f in test_paths] if test_paths else []

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=16) if val_datasets else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    best_val_loss, early_stop_counter = float('inf'), 0
    num_epochs, patience, burn_in_steps = 300, 300, 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_size, seq_len, _ = inputs.shape
            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()

            if burn_in_steps > 0 and seq_len > burn_in_steps:
                _, hidden = model(inputs[:, :burn_in_steps], hidden)

            current_t_min = inputs[:, 0, 1]  # T_outer
            current_input_temp = inputs[:, 0, 2]  # Input Temperature
            current_t_ave = inputs[:, 0, 3]  # T_avg
            time_weights = torch.linspace(1, 0, seq_len, device=device)
            batch_loss = 0.0

            for t in range(seq_len):
                input_t = torch.stack([inputs[:, t, 0], current_t_min, current_input_temp, current_t_ave],
                                      dim=1).unsqueeze(1)
                delta, hidden = model(input_t, hidden)
                # Now output is [T_inner, T_outer, T_avg] (3 values)
                output = delta[:, 0]  # [T_inner, T_outer, T_avg] (3 values)
                loss_t = weighted_loss(output, targets[:, t], time_weights=time_weights[t:t + 1])
                batch_loss += loss_t
                if t < seq_len - 1:
                    use_teacher = (torch.rand(batch_size, device=device) < 0.5).float()
                    ground_truth = inputs[:, t + 1, 1:4]  # [T_outer, Input_Temp, T_avg]
                    # output is [T_inner, T_outer, T_avg]
                    current_t_min = use_teacher * ground_truth[:, 0] + (1 - use_teacher) * output[
                        :, 1]  # T_outer from output[1]
                    current_input_temp = inputs[:, t + 1, 2]  # Use next Input Temperature
                    current_t_ave = use_teacher * ground_truth[:, 2] + (1 - use_teacher) * output[
                        :, 2]  # T_avg from output[2]

            (batch_loss / seq_len).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += batch_loss.item() / seq_len

        # —— 验证阶段 ——
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_batch in val_loader:
                    inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in val_batch]
                    seq_len = inputs.size(1)
                    hidden = model.init_hidden(inputs.size(0))
                    if burn_in_steps > 0 and seq_len > burn_in_steps:
                        _, hidden = model(inputs[:, :burn_in_steps], hidden)
                    current_t_min = inputs[:, 0, 1]
                    current_input_temp = inputs[:, 0, 2]
                    current_t_ave = inputs[:, 0, 3]
                    batch_val_loss = 0.0
                    for t in range(seq_len):
                        input_t = torch.stack([inputs[:, t, 0], current_t_min, current_input_temp, current_t_ave],
                                              dim=1).unsqueeze(1)
                        delta, hidden = model(input_t, hidden)
                        output = delta[:, 0]
                        loss_t = weighted_loss(output, targets[:, t])
                        batch_val_loss += loss_t
                        if t < seq_len - 1:
                            current_t_min = inputs[:, t + 1, 1]
                            current_input_temp = inputs[:, t + 1, 2]
                            current_t_ave = inputs[:, t + 1, 3]
                    val_loss += batch_val_loss.item() / seq_len
            ave_val_loss = val_loss / len(val_loader)
            scheduler.step(ave_val_loss)
        else:
            ave_val_loss = total_train_loss / max(1, len(train_loader))
            scheduler.step(ave_val_loss)

        print(f"[Epoch {epoch + 1}] Train Loss: {total_train_loss / len(train_loader):.4f}, "
              f"Val Loss: {ave_val_loss:.4f}")

        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, "best_gru.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(os.path.join(script_dir, "best_gru.pth")))
    return model, test_datasets


# === Testing function ===
def test_model(model, test_datasets):
    if model is None or not test_datasets:
        print("Model or dataset is empty");
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "GRU_without_burn_in")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTesting on {len(test_datasets)} test cases...")
    for idx, dataset in enumerate(test_datasets):
        file = os.path.basename(dataset.file_name)
        print(f"Processing test case {idx + 1}/{len(test_datasets)}: {file}")
        x, _, _, ft, ft_min, ft_max, ft_ave = dataset[0]
        x = x.to(device)
        hidden = model.init_hidden(1)

        # 只使用第一个时间步的实际温度值作为初始条件
        current_t_min = x[0, 1]  # 初始 T_outer
        current_t_ave = x[0, 3]  # 初始 T_avg
        preds = []

        for t in range(x.shape[0]):
            # 只使用 Input Temperature 作为输入（从数据中读取）
            current_input_temp = x[t, 2]  # 每个时间步的 Input Temperature

            input_t = torch.tensor([[x[t, 0], current_t_min, current_input_temp, current_t_ave]],
                                   dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                delta, hidden = model(input_t, hidden)
                output = delta[0, 0]  # [T_inner, T_outer, T_avg]
                pred = output.cpu().numpy()
                preds.append(pred)

                # 完全使用预测值更新（不再使用实际值）
                current_t_min = pred[1]  # 使用预测的 T_outer
                current_t_ave = pred[2]  # 使用预测的 T_avg
                # current_input_temp 会在下一次循环从 x[t+1, 2] 读取

        pred_seq = np.array(preds)

        # Actual values: ft, ft_min (T_outer), ft_max (T_inner), ft_ave (T_avg)
        # Need to add Input Temperature for complete scaler input
        # Note: pred_seq corresponds to predictions at time indices 1 to len(pred_seq)
        # Use ft[1:] to match prediction length
        dummy_actual = np.zeros((len(pred_seq), 5))
        dummy_actual[:, 0] = ft[1:len(pred_seq) + 1]  # Time (from index 1 to len(pred_seq))
        dummy_actual[:, 1] = ft_min[1:len(pred_seq) + 1]  # T_outer
        dummy_actual[:, 2] = ft_max[1:len(pred_seq) + 1]  # T_inner
        dummy_actual[:, 3] = ft_ave[1:len(pred_seq) + 1]  # T_avg

        # Need to add Input Temperature - get from x (input data)
        # x has columns: Time, T_outer, Input_Temp, T_avg
        dummy_actual[:, 4] = x[:, 2].cpu().numpy()  # Input Temperature

        # Predictions: pred_seq[:, 0]=T_inner, pred_seq[:, 1]=T_outer, pred_seq[:, 2]=T_avg
        # pred_seq corresponds to predictions at time indices 1 to n-1
        dummy_pred = np.zeros((len(pred_seq), 5))
        dummy_pred[:, 0] = ft[1:len(pred_seq) + 1]  # Time (from index 1 to n-1)
        dummy_pred[:, 1] = pred_seq[:, 1]  # T_outer prediction
        dummy_pred[:, 2] = pred_seq[:, 0]  # T_inner prediction
        dummy_pred[:, 3] = pred_seq[:, 2]  # T_avg prediction
        dummy_pred[:, 4] = x[:, 2].cpu().numpy()  # Input Temperature (use actual)

        inv_pred = dataset.scaler.inverse_transform(dummy_pred)
        inv_actual = dataset.scaler.inverse_transform(dummy_actual)

        # Calculate MAE for each temperature
        mae_inner = np.mean(np.abs(inv_actual[:, 2] - inv_pred[:, 2]))
        mae_outer = np.mean(np.abs(inv_actual[:, 1] - inv_pred[:, 1]))
        mae_avg = np.mean(np.abs(inv_actual[:, 3] - inv_pred[:, 3]))

        plt.plot(inv_actual[:, 0], inv_actual[:, 1], label="T_outer Actual", color="blue")
        plt.plot(inv_pred[:, 0], inv_pred[:, 1], "--", label="T_outer Pred", color="blue")
        plt.plot(inv_actual[:, 0], inv_actual[:, 3], label="T_avg Actual", color="green")
        plt.plot(inv_pred[:, 0], inv_pred[:, 3], "--", label="T_avg Pred", color="green")
        plt.plot(inv_actual[:, 0], inv_actual[:, 2], label="T_inner Actual", color="red")
        plt.plot(inv_pred[:, 0], inv_pred[:, 2], "--", label="T_inner Pred", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (C)")
        plt.title(f"Prediction - {file}")
        plt.legend()

        # Add MAE text to the plot
        mae_text = f"MAE: T_inner={mae_inner:.3f}°C, T_outer={mae_outer:.3f}°C, T_avg={mae_avg:.3f}°C"
        plt.text(0.02, 0.98, mae_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
        plt.tight_layout()

        # Save plot instead of showing
        output_path = os.path.join(output_dir, f"plot_{file}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to avoid displaying
        print(f"  -> Saved to: {output_path}")

    print(f"\nAll plots saved to: {output_dir}")


# === Main entry ===
if __name__ == "__main__":
    model, test_sets = train_model()
    if model is None or not test_sets:
        print("Model or dataset is empty. Exit.");
        raise SystemExit(1)
    test_model(model, test_sets)
