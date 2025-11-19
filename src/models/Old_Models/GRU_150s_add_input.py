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


# === Updated Dataset class ===
class UpdatedThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name

        # Check and normalize column names
        print(f"Processing file: {csv_file}")
        print(f"Original columns: {list(df.columns)}")

        # Column name normalization mapping
        column_mapping = {}

        # Handle inconsistency between T_ave vs T_avg
        if 'T_ave (C)' in df.columns and 'T_avg (C)' not in df.columns:
            column_mapping['T_ave (C)'] = 'T_avg (C)'
        elif 'T_avg (C)' in df.columns and 'T_ave (C)' not in df.columns:
            pass  # Already standardized
        elif 'T_ave (C)' in df.columns and 'T_avg (C)' in df.columns:
            print(f"Warning: Both T_ave and T_avg exist, using T_avg")

        # Apply column renaming
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"Column standardized: {column_mapping}")

        # Check required columns
        expected_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")

            # If Input Temperature column does not exist, use T_inner as a substitute
            if 'Input Temperature (C)' not in df.columns and 'T_inner (C)' in df.columns:
                df['Input Temperature (C)'] = df['T_inner (C)']
                print(f"Using T_inner as substitute for Input Temperature")
                if 'Input Temperature (C)' in missing_cols:
                    missing_cols.remove('Input Temperature (C)')

        # If key columns are still missing, skip this file
        if missing_cols:
            raise ValueError(f"File {csv_file} is missing required columns: {missing_cols}")

        print(f"Final columns: {list(df.columns)}")

        # Columns for normalization (external conditions + internal states)
        columns_for_scaling = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler

        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])

        # Process grouped by file
        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.external_conditions = []  # Store external conditions
        self.full_data = []

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
            self.full_data.append(group[columns_for_scaling].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.external_conditions = torch.tensor(np.array(self.external_conditions), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.full_data = np.array(self.full_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
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

    train_dir = os.path.join(data_dir, "150s Time steps")
    test_dir = os.path.join(data_dir, "test")

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
                print(f"Processing training file: {os.path.basename(path)}")

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
                    print(f"  ✓ File valid")
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
    train_datasets, val_datasets, test_datasets = [], [], []

    for path in actual_train_paths:
        try:
            dataset = UpdatedThermalDataset(path, scaler=scaler)
            train_datasets.append(dataset)
        except Exception as e:
            print(f"Failed to create training dataset {path}: {e}")

    for path in val_paths:
        try:
            dataset = UpdatedThermalDataset(path, scaler=scaler)
            val_datasets.append(dataset)
        except Exception as e:
            print(f"Failed to create validation dataset {path}: {e}")

    for path in test_paths:
        try:
            dataset = UpdatedThermalDataset(path, scaler=scaler)
            test_datasets.append(dataset)
        except Exception as e:
            print(f"Failed to create test dataset {path}: {e}")

    if not train_datasets:
        print("No valid training datasets")
        return None, None

    print(f"Valid datasets: Train {len(train_datasets)}, Val {len(val_datasets)}, Test {len(test_datasets)}")

    # DataLoaders
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=8, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=8) if val_datasets else None

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UpdatedThermalGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

    print(f"Using device: {device}")
    print("Start training...")

    # Training parameters
    num_epochs = 300
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

        # Validation phase
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
            val_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')

        # Progress log
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        if epoch % 10 == 0:
            print(f"[Epoch {epoch + 1:3d}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, "updated_best_gru.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if os.path.exists(os.path.join(script_dir, "updated_best_gru.pth")):
        model.load_state_dict(torch.load(os.path.join(script_dir, "updated_best_gru.pth")))
        print("Best model loaded")

    return model, test_datasets


# === Testing function ===
def test_updated_model(model, test_datasets):
    if not model or not test_datasets:
        print("Model or dataset is empty")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            # Inverse scaling for visualization
            dummy_pred = np.zeros((len(pred_array), 5))
            dummy_pred[:, 0] = full_data[1:len(pred_array) + 1, 0]  # Time
            dummy_pred[:, 1] = pred_array[:, 0]  # T_outer prediction
            dummy_pred[:, 2] = pred_array[:, 1]  # T_inner prediction
            dummy_pred[:, 3] = pred_array[:, 2]  # T_avg prediction
            dummy_pred[:, 4] = full_data[1:len(pred_array) + 1, 4]  # Input Temperature

            inv_pred = dataset.scaler.inverse_transform(dummy_pred)
            inv_actual = dataset.scaler.inverse_transform(full_data)

            # Plot results
            plt.figure(figsize=(14, 8))

            # Top subplot: temperature predictions
            plt.subplot(2, 1, 1)
            plt.plot(inv_actual[:, 0], inv_actual[:, 1], 'b-', label='T_outer Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 2], 'r-', label='T_inner Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 3], 'g-', label='T_avg Actual', linewidth=2)

            pred_times = inv_actual[1:len(inv_pred) + 1, 0]
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

            plt.tight_layout()
            plt.show()

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
        model, test_datasets = train_updated_model()

        if model and test_datasets:
            print("Training finished, start testing...")
            test_updated_model(model, test_datasets)
        else:
            print("Training failed")

    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()
