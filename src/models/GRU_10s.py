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

# Testing different dropout rates
configs = [
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 100, "dropout": 0.0},
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 100, "dropout": 0.1},
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 100, "dropout": 0.2},
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 100, "dropout": 0.3},
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 100, "dropout": 0.4},
]


# === Dataset class ===
class ThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name
        columns_for_scaling = ['Time (s)',
                               'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)'
                               ]
        rename_map = {
            "T_ave (C)": "T_avg (C)",
        }
        df.rename(columns=rename_map, inplace=True)
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler
        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])

        # === Add derived delta features ===
        df["dT_outer (C)"] = df["T_outer (C)"].diff().fillna(0)
        df["dT_avg (C)"] = df["T_avg (C)"].diff().fillna(0)
        df["dInput Temperature (C)"] = df["Input Temperature (C)"].diff().fillna(0)

        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.full_time, self.full_t_min, self.full_t_max, self.full_t_ave = [], [], [], []

        for _, group in grouped:
            # Input: Time, T_outer, Input Temperature, T_avg
            X_seq = group[["Time (s)",
                           "T_outer (C)", "Input Temperature (C)", "T_avg (C)",
                           "dT_outer (C)", "dInput Temperature (C)", "dT_avg (C)"]].values[:-1]
            # Output: T_inner, T_outer, T_avg
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
    def __init__(self, input_size=7, hidden_size=128, output_size=3, num_layers=5, dropout=0.1):
        super(ThermalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
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
def train_model(max_epochs=100, hidden_size=128, num_layers=5, dropout=0.1):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    possible_data_dirs = [
        os.path.join(script_dir, "..", "..", "data"),
        os.path.join(script_dir, "data"),
        "data"
    ]

    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break

    if not data_dir:
        print("Data directory not found")
        return None, None, float('inf'), [], []

    # Check for nested structure (local environment)
    if os.path.exists(os.path.join(data_dir, "fixed")):
        train_dir = os.path.join(data_dir, "fixed", "data", "data_in_10s")
        test_dir = os.path.join(data_dir, "fixed", "test_with_inputs", "test_in_10s")
    else:
        # Flat structure (Colab or original assumption)
        train_dir = os.path.join(data_dir, "data_in_10s")
        test_dir = os.path.join(data_dir, "test_in_10s")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    print(f"Found training files: {len(train_paths)}")
    print(f"Found test files: {len(test_paths)}")

    if not train_paths:
        print("No training files found")
        return None, None, float('inf'), [], []

    # === Split out validation set (5%) ===
    val_split = int(0.05 * len(train_paths))
    val_paths = train_paths[:val_split]
    actual_train_paths = train_paths[val_split:]

    # === Fit a unified scaler on training data ===
    train_dfs = [pd.read_csv(f) for f in actual_train_paths]
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_dfs)[["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)"]])

    train_datasets = [ThermalDataset(f, scaler=scaler) for f in actual_train_paths]
    val_datasets = [ThermalDataset(f, scaler=scaler) for f in val_paths]
    test_datasets = [ThermalDataset(f, scaler=scaler) for f in test_paths]

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=16)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = ThermalGRU(
        input_size=7,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    best_val_loss, early_stop_counter = float('inf'), 0
    patience, burn_in_steps = 100, 30

    # Lists to store loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_size, seq_len, _ = inputs.shape

            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()

            if burn_in_steps > 0:
                burn_inputs = inputs[:, 0:1, :].expand(-1, burn_in_steps, -1).clone()
                burn_inputs[:, :, 0] = inputs[:, :burn_in_steps, 0]
                _, hidden = model(burn_inputs, hidden)

            # Calculate scaled time step (assuming linear spacing)
            # Use the gap between step 0 and step 1 as the delta
            dt_scaled = inputs[:, 1, 0] - inputs[:, 0, 0]
            # Reshape for broadcasting if needed, though (batch,) matches behavior
            # Add scaled burn-in time
            burn_in_time_scaled = dt_scaled * burn_in_steps

            current_t_min = inputs[:, 0, 1]  # T_outer
            current_input_temp = inputs[:, 0, 2]  # Input Temperature
            current_t_ave = inputs[:, 0, 3]  # T_avg

            time_weights = torch.linspace(1, 0, seq_len, device=device)
            batch_loss = 0.0

            for t in range(0, seq_len):
                # Add scaled burn-in time to the input time channel
                t_input = inputs[:, t, 0] + burn_in_time_scaled
                
                input_t = torch.stack([t_input, current_t_min, current_input_temp, current_t_ave,
                                       inputs[:, t, 4],  # dT_outer
                                       inputs[:, t, 5],  # dInput
                                       inputs[:, t, 6],  # dT_avg
                                       ], dim=1).unsqueeze(1)

                delta, hidden = model(input_t, hidden)
                output = delta[:, 0]  # [T_inner, T_outer, T_avg]

                loss_t = weighted_loss(output, targets[:, t], time_weights=time_weights[t:t + 1])
                batch_loss += loss_t

                # Teacher forcing
                if t < seq_len - 1:
                    use_teacher = (torch.rand(batch_size, device=device) < 0.5).float()
                    ground_truth = inputs[:, t + 1, 1:4]  # [T_outer, Input_Temp, T_avg]
                    current_t_min = use_teacher * ground_truth[:, 0] + (1 - use_teacher) * output[:, 1]
                    current_input_temp = inputs[:, t + 1, 2]
                    current_t_ave = use_teacher * ground_truth[:, 2] + (1 - use_teacher) * output[:, 2]

            (batch_loss / seq_len).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += batch_loss.item() / seq_len

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_loader:
                inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in val_batch]
                batch_size, seq_len, _ = inputs.shape

                hidden = model.init_hidden(inputs.size(0))

                if burn_in_steps > 0:
                    burn_inputs = inputs[:, 0:1, :].expand(-1, burn_in_steps, -1).clone()
                    burn_inputs[:, :, 0] = inputs[:, :burn_in_steps, 0]
                    _, hidden = model(burn_inputs, hidden)

                dt_scaled = inputs[:, 1, 0] - inputs[:, 0, 0]
                burn_in_time_scaled = dt_scaled * burn_in_steps

                current_t_min = inputs[:, 0, 1]
                current_input_temp = inputs[:, 0, 2]
                current_t_ave = inputs[:, 0, 3]
                batch_val_loss = 0

                for t in range(0, seq_len):
                    t_input = inputs[:, t, 0] + burn_in_time_scaled
                    input_t = torch.stack([t_input, current_t_min, current_input_temp, current_t_ave,
                                           inputs[:, t, 4],
                                           inputs[:, t, 5],
                                           inputs[:, t, 6],
                                           ], dim=1).unsqueeze(1)
                    delta, hidden = model(input_t, hidden)

                    output = delta[:, 0]
                    loss_t = weighted_loss(output, targets[:, t])

                    batch_val_loss += loss_t

                    if t < seq_len - 1:
                        current_t_min = inputs[:, t + 1, 1]
                        current_input_temp = inputs[:, t + 1, 2]
                        current_t_ave = inputs[:, t + 1, 3]
                val_loss += batch_val_loss.item() / seq_len

        ave_train_loss = total_train_loss / len(train_loader)
        ave_val_loss = val_loss / len(val_loader)

        # Record loss history
        train_loss_history.append(ave_train_loss)
        val_loss_history.append(ave_val_loss)

        scheduler.step(ave_val_loss)
        print(f"[Epoch {epoch + 1}] Train Loss: {ave_train_loss:.4f}, Val Loss: {ave_val_loss:.4f}")

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
    return model, test_datasets, best_val_loss, train_loss_history, val_loss_history


# === Testing function ===
def test_model(model, test_datasets, burn_in_steps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "plot_GRU_10s_with_burn_in")
    os.makedirs(output_dir, exist_ok=True)

    for idx, dataset in enumerate(test_datasets):
        file = os.path.basename(dataset.file_name)
        x, _, _, ft, ft_min, ft_max, ft_ave = dataset[0]
        x = x.to(device)
        seq_len = x.shape[0]

        hidden = model.init_hidden(1)
        burn_in_preds = []
        if burn_in_steps > 0:
            # Run step-by-step burn-in to capture predictions
            # Use duplicated step 0 with incrementing time
            dt_scaled = x[1, 0] - x[0, 0]
            
            # Initial conditions from step 0
            current_t_min = x[0, 1]
            current_input_temp = x[0, 2]
            current_t_ave = x[0, 3]

            for t_step in range(burn_in_steps):
                time_val = t_step * dt_scaled # Time starts at 0 relative to sequence start? 
                # Yes, burn in is 0, 10... 290. Main sequence starts effectively at 300.
                
                # Input vector shape (1, 1, 7) for single step
                # Features: Time, T_outer, Input, T_avg, dT...
                # We duplicate x[0] features but update time
                input_t = x[0:1].clone() # Shape (1, 7)
                input_t[0, 0] = x[0, 0] + time_val # Update time
                input_t = input_t.unsqueeze(0) # (1, 1, 7)
                
                with torch.no_grad():
                     delta, hidden = model(input_t, hidden)
                     output = delta[0, 0]
                     burn_in_preds.append(output.cpu().numpy())
                     
                     # Update feedback loop for burn-in?
                     # The request is "burn_in of dups of timestep 0". 
                     # So inputs should stay constant x[0] except time.
                     # We DON'T update current_* from output for the NEXT input in burn-in,
                     # because that would be autoregressive simulation starting from constant.
                     # The user said "burn_in of dups of timestep 0".
                     # So inputs are fixed to x[0].
                     pass

        dt_scaled = x[1, 0] - x[0, 0]
        burn_in_time_scaled = dt_scaled * burn_in_steps
        
        start_t = 0 # Testing full sequence from 0

        current_t_min = x[start_t, 1]
        current_input_temp = x[start_t, 2]
        current_t_ave = x[start_t, 3]

        preds = []

        for t in range(start_t, seq_len):
            t_input = x[t, 0] + burn_in_time_scaled
            input_t = torch.tensor([[
                t_input,
                current_t_min,
                current_input_temp,
                current_t_ave,
                x[t, 4],
                x[t, 5],
                x[t, 6],
            ]], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                delta, hidden = model(input_t, hidden)
                output = delta[0, 0]

                pred = output.cpu().numpy()
                preds.append(pred)
                if t < x.shape[0] - 1:
                    current_t_min = pred[1]
                    current_input_temp = x[t + 1, 2]
                    current_t_ave = pred[2]

        # === Construct prediction arrays for plotting ===
        pred_seq = np.array(preds)  # Main predictions: shape (seq_len, 3) = [T_inner, T_outer, T_avg]
        burn_in_seq = np.array(burn_in_preds)  # Burn-in predictions: shape (burn_in_steps, 3)

        # Prepend known t=0 anchor to main predictions
        # ft_max = scaled T_inner, x[:,1] = scaled T_outer, x[:,3] = scaled T_avg
        known_t0 = np.array([[ft_max[start_t], x[start_t, 1].cpu().item(), x[start_t, 3].cpu().item()]])
        main_pred_with_anchor = np.concatenate([known_t0, pred_seq], axis=0)  # shape (seq_len+1, 3)

        # Combine burn-in + main predictions
        if len(burn_in_seq) > 0:
            all_pred = np.concatenate([burn_in_seq, main_pred_with_anchor], axis=0)
        else:
            all_pred = main_pred_with_anchor

        total_len = len(all_pred)

        # === Construct time axis (unscaled, for plotting) ===
        # ft = full_time = scaled time values for the FULL original sequence (length N)
        # We need to inverse_transform ft to get real seconds.
        # Scaler columns: [Time, T_outer, T_inner, T_avg, Input_Temp]
        # To inverse_transform just Time, we pad with zeros for the other 4 columns.
        ft_for_inv = np.zeros((len(ft), 5))
        ft_for_inv[:, 0] = ft
        real_time = dataset.scaler.inverse_transform(ft_for_inv)[:, 0]  # Unscaled time in seconds

        original_dt = real_time[1] - real_time[0]  # Should be ~10s
        burn_in_duration = burn_in_steps * original_dt

        # Burn-in time: 0, 10, 20, ... 290
        t_burn_in = np.arange(burn_in_steps) * original_dt
        # Main time: shifted by burn-in duration
        # main_pred_with_anchor has (seq_len+1) points, matching ft[0:seq_len+1]
        # But ft has N elements (full sequence), and seq_len = N-1 (x has N-1 rows)
        # So main_pred_with_anchor has N points, matching ft[0:N] = all of ft
        t_main = real_time[start_t : start_t + len(main_pred_with_anchor)] + burn_in_duration

        plot_time = np.concatenate([t_burn_in, t_main])

        # === Construct "Actual" arrays (scaled, for inverse_transform) ===
        dummy_actual = np.zeros((total_len, 5))
        dummy_actual[:, 0] = plot_time  # Will be overwritten by inverse_transform anyway

        # Burn-in "actual": constant initial values
        val_outer = x[0, 1].cpu().item()
        val_inner = ft_max[0]
        val_avg = x[0, 3].cpu().item()
        val_input = x[0, 2].cpu().item()

        if burn_in_steps > 0:
            dummy_actual[:burn_in_steps, 1] = val_outer
            dummy_actual[:burn_in_steps, 2] = val_inner
            dummy_actual[:burn_in_steps, 3] = val_avg
            dummy_actual[:burn_in_steps, 4] = val_input

        # Main "actual": from dataset
        # ft_min (T_outer), ft_max (T_inner), ft_ave (T_avg) have length N (full sequence)
        # main_pred_with_anchor has length N (seq_len+1 = N)
        # x has length N-1, so for Input Temp we need ft values instead
        n_main = len(main_pred_with_anchor)
        dummy_actual[burn_in_steps:, 1] = ft_min[start_t : start_t + n_main]
        dummy_actual[burn_in_steps:, 2] = ft_max[start_t : start_t + n_main]
        dummy_actual[burn_in_steps:, 3] = ft_ave[start_t : start_t + n_main]
        # For Input Temp: x has N-1 rows, but we need N values
        # Use x[:, 2] for first N-1 values, repeat last value for the Nth
        input_temp_scaled = x[:, 2].cpu().numpy()
        input_temp_padded = np.append(input_temp_scaled, input_temp_scaled[-1])
        dummy_actual[burn_in_steps:, 4] = input_temp_padded[start_t : start_t + n_main]

        # === Construct "Predicted" arrays (scaled, for inverse_transform) ===
        dummy_pred = np.zeros((total_len, 5))
        dummy_pred[:, 0] = plot_time
        dummy_pred[:, 1] = all_pred[:, 1]  # T_outer (index 1 in model output)
        dummy_pred[:, 2] = all_pred[:, 0]  # T_inner (index 0 in model output)
        dummy_pred[:, 3] = all_pred[:, 2]  # T_avg   (index 2 in model output)
        dummy_pred[:, 4] = dummy_actual[:, 4]  # Input Temp (same as actual)

        # === Inverse transform to get real temperatures ===
        inv_pred = dataset.scaler.inverse_transform(dummy_pred)
        inv_actual = dataset.scaler.inverse_transform(dummy_actual)

        # Overwrite time column with our constructed plot_time (already in real seconds)
        inv_pred[:, 0] = plot_time
        inv_actual[:, 0] = plot_time

        # === Calculate MAE on MAIN sequence only (after burn-in) ===
        mae_inner = np.mean(np.abs(inv_actual[burn_in_steps:, 2] - inv_pred[burn_in_steps:, 2]))
        mae_outer = np.mean(np.abs(inv_actual[burn_in_steps:, 1] - inv_pred[burn_in_steps:, 1]))
        mae_avg = np.mean(np.abs(inv_actual[burn_in_steps:, 3] - inv_pred[burn_in_steps:, 3]))

        # === Plot (main sequence only, skip burn-in) ===
        pa = inv_actual[burn_in_steps:]
        pp = inv_pred[burn_in_steps:]

        plt.figure(figsize=(12, 6))

        plt.plot(pa[:, 0], pa[:, 1], label="T_outer Actual", color="blue", alpha=0.6)
        plt.plot(pp[:, 0], pp[:, 1], "--", label="T_outer Pred", color="blue")

        plt.plot(pa[:, 0], pa[:, 3], label="T_avg Actual", color="green", alpha=0.6)
        plt.plot(pp[:, 0], pp[:, 3], "--", label="T_avg Pred", color="green")

        plt.plot(pa[:, 0], pa[:, 2], label="T_inner Actual", color="red", alpha=0.6)
        plt.plot(pp[:, 0], pp[:, 2], "--", label="T_inner Pred", color="red")

        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Prediction - {file}")
        plt.legend()

        mae_text = f"MAE (Main): T_inner={mae_inner:.3f}°C, T_outer={mae_outer:.3f}°C, T_avg={mae_avg:.3f}°C"
        plt.text(0.02, 0.98, mae_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"plot_{file}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save plotted data to CSV
        csv_output_dir = os.path.join(script_dir, "GRU_10s_with_burn_in")
        os.makedirs(csv_output_dir, exist_ok=True)
        csv_data = pd.DataFrame({
            'Time (s)': plot_time,
            'T_outer Actual (C)': inv_actual[:, 1],
            'T_inner Actual (C)': inv_actual[:, 2],
            'T_avg Actual (C)': inv_actual[:, 3],
            'T_outer Pred (C)': inv_pred[:, 1],
            'T_inner Pred (C)': inv_pred[:, 2],
            'T_avg Pred (C)': inv_pred[:, 3],
        })
        csv_path = os.path.join(csv_output_dir, f"test_case({idx + 1}).csv")
        csv_data.to_csv(csv_path, index=False)


    print(f"\nAll plots saved to: {output_dir}")


# === Main entry ===
if __name__ == "__main__":
    best_val_loss = float("inf")
    best_model = None
    best_test_sets = None
    best_config = None

    # Store all results
    all_results = []

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"Running Config: {cfg}")
        print(f"{'=' * 60}")

        model, test_sets, val_mae, train_history, val_history = train_model(
            max_epochs=cfg["max_epochs"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        )

        # Store results
        all_results.append({
            'config': cfg,
            'train_loss': train_history,
            'val_loss': val_history,
            'final_val_loss': val_mae
        })

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_config = cfg
            best_model = model
            best_test_sets = test_sets

        print(f"Validation MAE for this configuration: {val_mae:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Best Config: {best_config}")
    print(f"Best MAE: {best_val_loss:.4f}")
    print(f"{'=' * 60}")

    # Plot loss curves for different dropout rates
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Plot training and validation loss
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for result in all_results:
        dropout = result['config']['dropout']
        plt.plot(result['train_loss'], label=f'Dropout={dropout}', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss vs Dropout Rate', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for result in all_results:
        dropout = result['config']['dropout']
        plt.plot(result['val_loss'], label=f'Dropout={dropout}', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss vs Dropout Rate', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curves_path = os.path.join(script_dir, 'dropout_comparison.png')
    plt.savefig(loss_curves_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss curves saved to: {loss_curves_path}")
    plt.close()

    # Plot final validation loss bar chart
    plt.figure(figsize=(10, 6))
    dropouts = [r['config']['dropout'] for r in all_results]
    final_losses = [r['final_val_loss'] for r in all_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(dropouts)))
    bars = plt.bar(range(len(dropouts)), final_losses, color=colors, edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(dropouts)), [f'{d}' for d in dropouts], fontsize=11)
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Final Validation Loss', fontsize=12)
    plt.title('Final Validation Loss for Different Dropout Rates', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (loss, bar) in enumerate(zip(final_losses, bars)):
        plt.text(i, loss + 0.001, f'{loss:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best config
    best_idx = final_losses.index(min(final_losses))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    bar_chart_path = os.path.join(script_dir, 'dropout_final_loss.png')
    plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
    print(f"Final loss comparison saved to: {bar_chart_path}")
    plt.close()

    # Print summary table
    print(f"\n{'=' * 60}")
    print("Summary of Results:")
    print(f"{'=' * 60}")
    print(f"{'Dropout':<12} {'Final Val Loss':<20} {'Epochs Run':<15}")
    print(f"{'-' * 60}")
    if best_config:
        for result in all_results:
            dropout = result['config']['dropout']
            final_loss = result['final_val_loss']
            epochs = len(result['val_loss'])
            marker = " <-- BEST" if dropout == best_config['dropout'] else ""
            print(f"{dropout:<12.2f} {final_loss:<20.6f} {epochs:<15}{marker}")
    else:
        print("No successful training runs completed.")
    print(f"{'=' * 60}\n")

    # Test the best model
    if best_model:
        print("Testing best model...")
        test_model(best_model, best_test_sets)
    else:
        print("Skipping testing as no model was trained.")