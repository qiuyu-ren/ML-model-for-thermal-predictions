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

# === Creating Cache ===
# - so model does not take as long to train (despite time windows)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# === Updated Dataset class ===
class UpdatedThermalDataset(Dataset):
    log_status = [] # list to collect success/fail info

    def __init__(self, csv_file, scaler=None, window_size=None, stride=None):
        self.file_name = os.path.basename(csv_file)

        # Check if cache exists
        CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Use a unique cache name per file + config
        cache_file = os.path.join(
            CACHE_DIR, f"{self.file_name}_w{window_size}_s{stride}.pt"
        )

        if os.path.exists(cache_file):
            try:
                # need to set weights_only = True to suppress warning
                cached = torch.load(cache_file, weights_only=False)
                self.X = cached["X"]
                self.Y = cached["Y"]
                self.external_conditions = cached["external"]
                self.time_values = cached["time"]
                self.full_data = cached["full"]
                self.scaler = cached["scaler"]
                # print(f"[CACHE] Loaded {self.file_name} from cache")
                return  # Skip CSV parsing and windowing
            except Exception as e:
                print(f"[CACHE] Failed to load cache for {self.file_name}: {e}")


        try:
            df = pd.read_csv(csv_file)
            df["FileName"] = self.file_name
            original_cols = list(df.columns)
            column_mapping = {} # Column name normalization mapping

            # Normalize T_ave & T_avg
            if 'T_ave (C)' in df.columns and 'T_avg (C)' not in df.columns:
                column_mapping['T_ave (C)'] = 'T_avg (C)'
            elif 'T_avg (C)' in df.columns and 'T_ave (C)' not in df.columns:
                pass  # Already standardized
            # Apply column renaming
            if column_mapping:
                df = df.rename(columns=column_mapping)

            # Check required columns
            expected_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
            missing_cols = [col for col in expected_cols if col not in df.columns]

            # If Input Temperature column does not exist, use T_inner as a substitute
            if 'Input Temperature (C)' not in df.columns and 'T_inner (C)' in df.columns:
                df['Input Temperature (C)'] = df['T_inner (C)']
                if 'Input Temperature (C)' in missing_cols:
                    missing_cols.remove('Input Temperature (C)')

            # If key columns are still missing, skip this file
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Normalize columns (external conditions + internal states)
            cols_for_scale = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
            self.scaler = scaler or MinMaxScaler().fit(df[cols_for_scale])
            df[cols_for_scale] = self.scaler.transform(df[cols_for_scale])

            # Group and store tensors
            grouped = df.groupby("FileName")
            self.X, self.Y, self.time_values, self.external_conditions, self.full_data = [], [], [], [], []

            # Process grouped by file
            for _, group in grouped:
                # External condition sequence: [Time, Input Temperature]
                external_seq = group[["Time (s)", "Input Temperature (C)"]].values
                # Input sequence: [Time, Input Temperature, T_outer, T_inner, T_avg] (current step)
                X_seq = group[["Time (s)", "Input Temperature (C)", "T_outer (C)", "T_inner (C)", "T_avg (C)"]].values[:-1]
                # Target sequence: [T_outer, T_inner, T_avg] (next step)
                Y_seq = group[["T_outer (C)", "T_inner (C)", "T_avg (C)"]].values[1:]
                time_vals = group["Time (s)"].values[1:]

                self.X.append(X_seq)
                self.Y.append(Y_seq)
                self.external_conditions.append(external_seq)
                self.time_values.append(time_vals)
                self.full_data.append(group[cols_for_scale].values)

            # Sliding Window Augmentation
            if window_size is not None and stride is not None:
                X_windows, Y_windows = [], []
                external_list = []
                time_list = []
                full_list = []

                for i, (x,y) in enumerate(zip(self.X, self.Y)):
                    # Create overlapping windows
                    xw, yw = self.create_windows(x, y, window_size, stride)
                    X_windows.extend(xw)
                    Y_windows.extend(yw)

                    # Replicate external/time/full data for each window
                    ext_seq = self.external_conditions[i]
                    time_seq = self.time_values[i]
                    full_seq = self.full_data[i]
                    total_steps = len(full_seq)

                    # Append metadata for each window
                    for w in range(len(xw)):
                        start = w * stride
                        end = start + window_size
                        external_list.append(ext_seq[start:end])
                        time_list.append(time_seq[start:end])
                        full_list.append(full_seq[start:end])

                # Replace X, Y with windowed Data & Convert to same length containers
                self.X = torch.tensor(np.array(X_windows), dtype=torch.float32)
                self.Y = torch.tensor(np.array(Y_windows), dtype=torch.float32)
                self.external_conditions = torch.tensor(np.array(external_list), dtype=torch.float32)
                self.time_values = np.array(time_list)
                self.full_data = np.array(full_list)

                # Debugging
                #print(f"[DEBUG] After windowing: X={len(self.X)}, ext={len(self.external_conditions)}, full={len(self.full_data)}")
            else:
                # No windowing - single sequence per file
                self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
                self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
                self.external_conditions = torch.tensor(np.array(self.external_conditions), dtype=torch.float32)
                self.time_values = np.array(self.time_values)
                self.full_data = np.array(self.full_data)
            # Success
            UpdatedThermalDataset.log_status.append(
                (self.file_name, "OK", original_cols, list(df.columns))
            )

            # save to cache
            try:
                torch.save({
                    "X": self.X,
                    "Y": self.Y,
                    "external": self.external_conditions,
                    "time": self.time_values,
                    "full": self.full_data,
                    "scaler": self.scaler
                }, cache_file)
                # print(f"[CACHE] Saved {self.file_name} to cache")
            except Exception as e:
                print(f"[CACHE] Failed to save cache for {self.file_name}: {e}")


        except Exception as e:
            # Failure
            UpdatedThermalDataset.log_status.append(
                (self.file_name, f"Failed: {str(e)}", None, None)
            )
            raise
    # helper to create sliding windows
    def create_windows(self, X_seq, Y_seq, window_size, stride):
        """Split the data sequence into overlapping windows"""
        X_list, Y_list = [], []
        total_steps = len(X_seq)
        for start in range(0, total_steps - window_size + 1, stride):
            end = start + window_size
            X_list.append(X_seq[start:end])
            Y_list.append(Y_seq[start:end])
        return X_list, Y_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        assert len(self.X) == len(self.Y) == len(self.external_conditions) == len(self.full_data), \
            f"Len mismatch: X={len(self.X)}, Y={len(self.Y)}, ext={len(self.external_conditions)}, full={len(self.full_data)}"
        return (
            self.X[idx], self.Y[idx],
            self.external_conditions[idx],
            self.time_values[idx],
            self.full_data[idx]
        )

# === Updated GRU model ===
class UpdatedThermalGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=3, num_layers=3):
        super(UpdatedThermalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # External condition encoder (process [Time, Input Temperature])
        self.external_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # State encoder (process [T_outer, T_inner, T_avg])
        self.state_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Main GRU
        self.gru = nn.GRU(64, hidden_size, num_layers, batch_first=True, dropout=0.1)

        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape

        # Split external conditions and state
        external = x[:, :, :2]  # [Time, Input Temperature]
        state = x[:, :, 2:]  # [T_outer, T_inner, T_avg]

        # Encode
        external_encoded = self.external_encoder(external)
        state_encoded = self.state_encoder(state)

        # Concatenate features
        combined = torch.cat([external_encoded, state_encoded], dim=-1)

        # GRU
        out, hidden = self.gru(combined, hidden)

        # Output prediction
        output = self.output_net(out)

        return output, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

# === Loss function ===
def thermal_loss(predictions, targets, temp_weights=torch.tensor([1.0, 1.0, 1.0])):
    """
    Custom loss for temperature prediction
    temp_weights: [weight for T_outer, T_inner, T_avg]
    """
    temp_weights = temp_weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * temp_weights
    return torch.mean(loss)

# === Training function ===
def train_updated_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Search for data directory
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
        return None, None

    # Toggle Burn in
    use_burn_in = False
    if use_burn_in:
        train_dir = r"C:\Users\jshih\OneDrive\Desktop\ML-model-for-thermal-predictions-max\data\data\data_in_10s_with_burn_in"
        test_dir = r"C:\Users\jshih\OneDrive\Desktop\ML-model-for-thermal-predictions-max\data\test\test_in_10s_with_burn_in"
        #train_dir = os.path.join(data_dir, "src/models/data_in_10s_with_burn_in")
        #test_dir = os.path.join(data_dir, "fixed", "/test/test_in_10s_with_burn_in")
    else:
        #train_dir = os.path.join(data_dir, "src/models/data_in_10s")
        #test_dir = os.path.join(data_dir, "fixed", "/test/test_in_10s")
        train_dir = r"C:\Users\jshih\OneDrive\Desktop\ML-model-for-thermal-predictions-max\data\data\data_in_10s"
        test_dir = r"C:\Users\jshih\OneDrive\Desktop\ML-model-for-thermal-predictions-max\data\test\test_in_10s"

    # Confirm Directories
    print(f" Using training data from: {train_dir}")
    print(f" Using testing data from:  {test_dir}")

    # Verify Directories
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f" Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f" Testing directory not found: {test_dir}")

    # Gather CSV files
    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    print(f"Found training files: {len(train_paths)}")
    print(f"Found test files: {len(test_paths)}")

    if not train_paths:
        print("No training files found")
        return None, None

    # Validation split
    val_split = max(1, int(0.1 * len(train_paths)))
    val_paths = train_paths[:val_split]
    actual_train_paths = train_paths[val_split:]

    # Create a unified scaler
    try:
        train_dfs = []
        for path in actual_train_paths:
            try:
                df = pd.read_csv(path)
                #print(f"Processing training file: {os.path.basename(path)}")

                # Standardize column names
                if 'T_ave (C)' in df.columns and 'T_avg (C)' not in df.columns:
                    df = df.rename(columns={'T_ave (C)': 'T_avg (C)'})
                    print(f"  Column standardized: T_ave -> T_avg")

                # Handle missing Input Temperature column
                if 'Input Temperature (C)' not in df.columns and 'T_inner (C)' in df.columns:
                    df['Input Temperature (C)'] = df['T_inner (C)']
                    print(f"  Using T_inner as Input Temperature")

                # Check required columns
                required_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
                if all(col in df.columns for col in required_cols):
                    train_dfs.append(df)
                    #print(f"  ✓ File valid")
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    print(f"  ✗ Skipped file, missing columns: {missing}")

            except Exception as e:
                print(f"  ✗ Error processing file: {e}")
                continue

        if not train_dfs:
            print("No valid training data files")
            return None, None

        scaler = MinMaxScaler()
        combined_df = pd.concat(train_dfs)
        scaler_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        scaler.fit(combined_df[scaler_cols])

        print(f"Scaler created successfully, using {len(train_dfs)} files")

    except Exception as e:
        print(f"Error creating scaler: {e}")
        return None, None

    # Create datasets
    failed_files = [] # collect (filename, error message) for reporting

    # Window size, Stride pair
    window_configs = [
        #(32, 16),
        (64, 32), # This is clearly the best sliding window parameters
        #(128, 64),
    ]

    # Store best (window_size, stride, best_val_loss)
    results = []

    for window_size, stride in window_configs:
        print(f"Training with window={window_size}, stride={stride}")

        # === Load datasets with current window parameters ===
        def load_datasets(file_paths, label, scaler):
            datasets = []
            for path in file_paths:
                try:
                    dataset = UpdatedThermalDataset(path, scaler=scaler, window_size =window_size, stride=stride)
                    datasets.append(dataset)
                except Exception as e:
                    failed_files.append((os.path.basename(path), str(e)))
            print(f"Loaded {len(datasets)}/{len(file_paths)} {label} files successfully.")
            return datasets

        train_datasets = load_datasets(actual_train_paths, "training", scaler)
        val_datasets = load_datasets(val_paths, "validation", scaler)
        test_datasets = load_datasets(test_paths, "test", scaler)

        # Summary
        if UpdatedThermalDataset.log_status:
            failed = [f for f in UpdatedThermalDataset.log_status if "FAILED" in f[1]]
            print(f"Failed files: {len(failed)}")

            if failed:
                print("Problematic files:")
                for f, msg, _, _ in failed:
                    print(f"{f}: {msg}")

        # Summary Message
        if failed_files:
            print("Some files could not be processed correctly")
            for fname, err in failed_files:
                print(f"{fname}: {err}")
        else:
            print(f"Successfully processed {len(train_datasets)} training files")

        # === DataLoaders ===
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=64,
            num_workers=4,
            pin_memory=True
        ) if val_datasets else None

        # Model and optimizer
        device = torch.device("cuda") # for mac / else use cuda for NVIDIA GPUs
        model = UpdatedThermalGRU().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

        print(f"Using device: {device}")
        print("Start training...")

        # Training parameters
        num_epochs = 150
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 30

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            num_batches = 0

            for batch in train_loader:
                try:
                    X, Y, external_conditions, _, _ = batch
                    X, Y = X.to(device), Y.to(device)
                    batch_size, seq_len = X.shape[0], X.shape[1]
                    hidden = model.init_hidden(batch_size)
                    optimizer.zero_grad()
                    # Forward
                    predictions, hidden = model(X, hidden)
                    # Loss
                    loss = thermal_loss(predictions, Y)
                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue

            avg_train_loss = total_train_loss / max(1, num_batches)
            # === Validation phase ===
            val_loss = 0
            if val_loader:
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        try:
                            X, Y, _, _, _ = val_batch
                            X, Y = X.to(device), Y.to(device)

                            hidden = model.init_hidden(X.size(0))
                            predictions, hidden = model(X, hidden)
                            loss = thermal_loss(predictions, Y)
                            val_loss += loss.item()

                        except Exception as e:
                            continue

                val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                scheduler.step(val_loss)
            else:
                val_loss = avg_train_loss

            # Progress logging
            if epoch % 10 == 0:
                print(f"[window:{window_size}/stride:{stride} [Epoch {epoch + 1:3d}] Train Loss: {avg_train_loss:.6f}, Val MAE: {val_loss:.6f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_window = window_size
                torch.save(model.state_dict(), os.path.join(script_dir, "updated_best_gru.pth"))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Store results
        results.append((window_size, stride, best_val_loss))
        print(f"Finished window={window_size}, stride={stride}, best_val_loss={best_val_loss:.6f}")

    for w, s, loss in results:
        print(f"Window: {w}, Stride: {s}, Best loss: {loss}")

    # === Return best model ===
    try:
        best_model = UpdatedThermalGRU()  # Recreate model structure
        best_model.load_state_dict(best_model_state)  # Load best weights
        best_model.to(device)
        best_model.eval()
        print("\n[INFO] Loaded best model based on validation loss.")
        return best_model, test_datasets, best_window
    except Exception as e:
        print(f"[WARN] Could not load best model: {e}")
        # otherwise return last trained model
        return model, test_datasets, best_window

# === Testing function ===
def test_updated_model(model, test_datasets, best_window):
    if not model or not test_datasets:
        print("Model or dataset is empty")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU via MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU only")

    model.eval()

    print(f"\nStart testing {len(test_datasets)} files...")

    for dataset in test_datasets:
        try:
            file_name = dataset.file_name
            X, Y, external_conditions, time_values, full_data = dataset[0]

            print(f"\n=== Test file: {file_name} ===")

            X = X.to(device)
            external_conditions = external_conditions.to(device)

            hidden = model.init_hidden(1)
            predictions = []

            # Current state
            current_state = X[0, 2:].clone()  # [T_outer, T_inner, T_avg]

            with torch.no_grad():
                for t in range(len(external_conditions)):
                    # Construct input: [Time, Input Temperature, T_outer, T_inner, T_avg]
                    if t < len(X):
                        input_t = torch.cat([
                            external_conditions[t],  # [Time, Input Temperature]
                            current_state  # [T_outer, T_inner, T_avg]
                        ]).unsqueeze(0).unsqueeze(0)

                        # Predict next state
                        pred, hidden = model(input_t, hidden)
                        pred_state = pred[0, 0]  # [T_outer, T_inner, T_avg]

                        predictions.append(pred_state.cpu().numpy())

                        # Update current state for next step
                        if t < len(X) - 1:
                            # Hybrid strategy: use ground truth for first few steps, then predictions
                            if t >= 3:
                                current_state = pred_state.clone()
                            else:
                                current_state = X[t + 1, 2:].clone()

            if not predictions:
                print("No predictions generated")
                continue

            pred_array = np.array(predictions)

            # === FIX ===
            usable_len = min(len(pred_array), full_data.shape[0] - 1)
            pred_array = pred_array[:usable_len]  # cut predictions if needed

            dummy_pred = np.zeros((usable_len, 5))
            dummy_pred[:, 0] = full_data[1:1 + usable_len, 0]  # safe, zero errors
            dummy_pred[:, 1] = pred_array[:, 0]  # T_outer prediction
            dummy_pred[:, 2] = pred_array[:, 1]  # T_inner prediction
            dummy_pred[:, 3] = pred_array[:, 2]  # T_avg prediction
            dummy_pred[:, 4] = full_data[1:1 + usable_len, 4]  # Input Temperature

            inv_pred = dataset.scaler.inverse_transform(dummy_pred)
            inv_actual = dataset.scaler.inverse_transform(full_data)

            # Plot results
            plt.figure(figsize=(14, 8))

            # Top subplot: temperature predictions
            plt.subplot(2, 1, 1)
            plt.plot(inv_actual[:, 0], inv_actual[:, 1], 'b-', label='T_outer Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 2], 'r-', label='T_inner Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 3], 'g-', label='T_avg Actual', linewidth=2)

            pred_times = inv_actual[1:1 + len(inv_pred), 0]
            plt.plot(pred_times, inv_pred[:, 1], 'b--', label='T_outer Pred', linewidth=2, alpha=0.8)
            plt.plot(pred_times, inv_pred[:, 2], 'r--', label='T_inner Pred', linewidth=2, alpha=0.8)
            plt.plot(pred_times, inv_pred[:, 3], 'g--', label='T_avg Pred', linewidth=2, alpha=0.8)

            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Temperature Prediction - {file_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            # Bottom subplot: input condition
            plt.subplot(2, 1, 2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 4], 'k-', label='Input Temperature', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Input Temperature (°C)')
            plt.title('External Input Condition')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # === Create plot output directory ===
            plot_dir = os.path.join(os.path.dirname(__file__), "plot_GRU_10s")
            os.makedirs(plot_dir, exist_ok=True)

            # Save filename: same as CSV name but PNG
            png_name = file_name.replace(".csv", ".png")
            png_path = os.path.join(plot_dir, png_name)

            # Save plot
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            print(f"[PLOT SAVED] {png_path}")

            plt.show()
            plt.close()

            # Compute error
            actual_temps = inv_actual[1:len(inv_pred) + 1, 1:4]  # [T_outer, T_inner, T_avg]
            pred_temps = inv_pred[:, 1:4]

            mae = np.mean(np.abs(actual_temps - pred_temps), axis=0)
            print(f"Mean Absolute Error:")
            print(f"  T_outer: {mae[0]:.2f}°C")
            print(f"  T_inner: {mae[1]:.2f}°C")
            print(f"  T_avg: {mae[2]:.2f}°C")

        except Exception as e:
            print(f"Error testing file {dataset.file_name}: {e}")
            import traceback
            traceback.print_exc()

# === Main function ===
if __name__ == "__main__":
    print("Start training thermal prediction model adapted for new data structure...")

    try:
        model, test_datasets, best_window = train_updated_model()

        if model and test_datasets:
            print("Training finished, start testing...")
            test_updated_model(model, test_datasets, best_window)
        else:
            print("Training failed")

    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()