import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# === 1. Physics Informed Loss Module ===
class PhysicsLoss(nn.Module):
    def __init__(self, scaler_min, scaler_scale):
        super(PhysicsLoss, self).__init__()
        self.d = 0.1016
        self.w = 0.00635
        self.rho = 2400.0
        self.x_raw = nn.Parameter(torch.tensor([0.0]))

        self.register_buffer("s_min", torch.tensor(scaler_min, dtype=torch.float32))
        self.register_buffer("s_scale", torch.tensor(scaler_scale, dtype=torch.float32))

    def unscale(self, Ti_s, To_s, Ta_s, Text_s):
        # Inverse of MinMaxScaler: X = (X_scaled - min_) / scale_
        Ti = (Ti_s - self.s_min[2]) / self.s_scale[2]
        To = (To_s - self.s_min[1]) / self.s_scale[1]
        Ta = (Ta_s - self.s_min[3]) / self.s_scale[3]
        Text = (Text_s - self.s_min[4]) / self.s_scale[4]
        return Ti, To, Ta, Text

    def get_alpha(self, T):
        k = torch.where(T <= 100, 2.5, torch.where(T >= 400, 2.0, 2.67 - 0.0014 * T))
        cp = torch.where(T <= 100, 900.0, torch.where(T <= 200, 800.0 + T,
             torch.where(T < 400, 900.0 + 0.5 * T, 1100.0)))
        return k / (self.rho * cp)

    def forward(self, preds, input_temps_scaled, dt=150.0): # dt set to match data stride
        Ti, To, Ta, Tin = self.unscale(preds[:, :, 0], preds[:, :, 1], preds[:, :, 2], input_temps_scaled)

        dTi_dt = (Ti[:, 1:] - Ti[:, :-1]) / dt
        dTo_dt = (To[:, 1:] - To[:, :-1]) / dt
        dTa_dt = (Ta[:, 1:] - Ta[:, :-1]) / dt

        Ti_c, To_c, Ta_c, Tin_c = Ti[:, 1:], To[:, 1:], Ta[:, 1:], Tin[:, 1:]
        alpha_i, alpha_o, alpha_a = self.get_alpha(Ti_c), self.get_alpha(To_c), self.get_alpha(Ta_c)

        safe_fraction = 0.1 + 0.8 * torch.sigmoid(self.x_raw)
        x = safe_fraction * self.d
        dist_outer = self.d - x
        dist_outer = self.d - x

        res_inner = dTi_dt - alpha_i * ((Tin_c - Ti_c) / (self.w**2))
        res_avg = dTa_dt - alpha_a * ((Ti_c - Ta_c) / (x**2) + (To_c - Ta_c) / (dist_outer**2))
        res_outer = dTo_dt - alpha_o * ((Ti_c - To_c) / (self.d**2) + (Ta_c - To_c) / (dist_outer**2))

        loss_ode = torch.mean(res_inner**2 + res_avg**2 + res_outer**2)
        penalty_term = torch.relu(torch.min(Ti_c, To_c) - Ta_c) + torch.relu(Ta_c - torch.max(Ti_c, To_c))
        loss_penalty = torch.mean(penalty_term**2)

        return loss_ode + loss_penalty

# === 2. Dataset with Full State Inputs ===
class ThermalWindowDataset(Dataset):
    def __init__(self, csv_paths, window_size=50):
        self.X, self.Y = [], []
        all_dfs = [pd.read_csv(p).rename(columns={"T_ave (C)": "T_avg (C)"}) for p in csv_paths]
        combined = pd.concat(all_dfs)
        cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        self.scaler = MinMaxScaler().fit(combined[cols])

        for df in all_dfs:
            data = self.scaler.transform(df[cols])
            for i in range(0, len(data) - window_size, 1):
                # Input Size 5: [Time, Ti, To, Ta, Input]
                self.X.append(data[i : i + window_size, [0, 2, 1, 3, 4]])
                # Target Size 3: [Ti, To, Ta]
                self.Y.append(data[i + 1 : i + window_size + 1, [2, 1, 3]])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        print(f"Dataset created: {len(self.X)} samples.")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# === 3. GRU Model (Input Size Updated) ===
class ThermalGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=3, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.fc(out), h

def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwd = os.getcwd()
    train_dir = os.path.join(cwd, "data", "train", "data_in_150s")
    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))

    dataset = ThermalWindowDataset(train_paths, window_size=50)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ThermalGRU(input_size=5).to(device)
    phys_fn = PhysicsLoss(dataset.scaler.min_, dataset.scaler.scale_).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': phys_fn.parameters(), 'lr': 0.01}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    lambda_phys, lambda_ic = 0.001, 1.0
    epochs = 300

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        total_d, total_p = 0, 0
        total_combined = 0  # --- ADDED: Track combined loss for scheduler ---

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            batch_preds = []
            h = None
            curr_input = bx[:, 0:1, :]

            for t in range(bx.size(1)):
                out, h = model(curr_input, h)
                batch_preds.append(out)
                if t < bx.size(1) - 1:
                    if random.random() < 0.3: # Scheduled Sampling
                        curr_input = torch.cat([bx[:, t+1:t+2, 0:1], out, bx[:, t+1:t+2, 4:5]], dim=-1)
                    else:
                        curr_input = bx[:, t+1:t+2, :]

            preds = torch.cat(batch_preds, dim=1)

            loss_data = torch.mean((preds - by)**2)
            loss_ic = torch.mean((preds[:, 0, :] - by[:, 0, :])**2)
            loss_phys = phys_fn(preds, bx[:, :, 4])

            total_loss = loss_data + (lambda_phys * loss_phys) + (lambda_ic * loss_ic)
            total_loss.backward()

            # Gradient clipping (from previous step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_d += loss_data.item()
            total_p += loss_phys.item()
            total_combined += total_loss.item() # --- ADDED ---

        # --- ADDED: Step the scheduler at the end of the epoch ---
        avg_epoch_loss = total_combined / len(train_loader)
        scheduler.step(avg_epoch_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            # --- ADDED: Get current Learning Rate for printing ---
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Data Loss: {total_d/len(train_loader):.6f} | Phys Loss: {total_p/len(train_loader):.6f} | LR: {current_lr:.6f}")
    learned_fraction = 0.1 + 0.8 * torch.sigmoid(phys_fn.x_raw).item()
    print(f"The PINN placed the T_avg sensor at {learned_fraction * 100:.2f}% of the wall thickness.")

    return model, dataset.scaler

# === 5. Testing with t=0 prepended ===
def test_pinn(model, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, "data", "test", "test_in_150s")
    test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))
    output_dir = os.path.join(cwd, "results", "PINN_Final")
    os.makedirs(output_dir, exist_ok=True)

    all_mae_inner = []
    all_mae_outer = []
    all_mae_avg = []

    # --- ADDED: CSV STORAGE ---
    csv_results = []

    cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']

    for path in test_paths:
        df = pd.read_csv(path).rename(columns={"T_ave (C)": "T_avg (C)"})
        scaled_data = scaler.transform(df[cols])

        initial_state = scaled_data[0, [2, 1, 3]]
        preds = [initial_state]

        curr_in = torch.tensor(scaled_data[0, [0, 2, 1, 3, 4]], dtype=torch.float32).view(1, 1, 5).to(device)
        h = None

        with torch.no_grad():
            for t in range(len(scaled_data) - 1):
                out, h = model(curr_in, h)
                p = out.cpu().squeeze().numpy()
                preds.append(p)
                curr_in = torch.tensor([scaled_data[t+1, 0], p[0], p[1], p[2], scaled_data[t+1, 4]],
                                       dtype=torch.float32).view(1, 1, 5).to(device)

        preds = np.array(preds)
        actuals = scaled_data

        def unscale(pred_arr, actual_arr):
            p_dummy, a_dummy = np.zeros((len(pred_arr), 5)), np.zeros((len(actual_arr), 5))
            p_dummy[:, [0, 2, 1, 3, 4]] = actual_arr[:, [0, 2, 1, 3, 4]]
            p_dummy[:, [2, 1, 3]] = pred_arr
            a_dummy[:, :] = actual_arr
            return scaler.inverse_transform(p_dummy), scaler.inverse_transform(a_dummy)

        inv_p, inv_a = unscale(preds, actuals)
        time = inv_a[:, 0]

        m_inner = np.mean(np.abs(inv_p[1:, 2] - inv_a[1:, 2]))
        m_outer = np.mean(np.abs(inv_p[1:, 1] - inv_a[1:, 1]))
        m_avg   = np.mean(np.abs(inv_p[1:, 3] - inv_a[1:, 3]))
        all_mae_inner.append(m_inner)
        all_mae_outer.append(m_outer)
        all_mae_avg.append(m_avg)

        # --- ADDED: APPEND TO CSV DATA ---
        csv_results.append({
            "File": os.path.basename(path),
            "Inner_MAE": m_inner,
            "Avg_MAE": m_avg,
            "Outer_MAE": m_outer,
            "Mean_MAE": (m_inner + m_outer + m_avg) / 3
        })

        plt.figure(figsize=(12, 7))
        plt.plot(time, inv_a[:, 4], 'k--', alpha=0.5, label="Input Temp (T_input)")
        plt.plot(time, inv_a[:, 2], 'r-', label=f"Actual Inner (MAE: {m_inner:.2f})")
        plt.plot(time, inv_p[:, 2], 'r--', label="Pred Inner")
        plt.plot(time, inv_a[:, 1], 'b-', label=f"Actual Outer (MAE: {m_outer:.2f})")
        plt.plot(time, inv_p[:, 1], 'b--', label="Pred Outer")
        plt.plot(time, inv_a[:, 3], 'g-', label=f"Actual Avg (MAE: {m_avg:.2f})")
        plt.plot(time, inv_p[:, 3], 'g--', label="Pred Avg")

        plt.title(f"File: {os.path.basename(path)}")
        plt.xlabel("Time (s)"); plt.ylabel("Temperature (°C)"); plt.legend(bbox_to_anchor=(1.05, 1)); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(path)}.png"))
        plt.close()

    # --- ADDED: SAVE CSV ---
    pd.DataFrame(csv_results).to_csv(os.path.join(output_dir, "test_mae_results.csv"), index=False)

    avg_inner = np.mean(all_mae_inner)
    avg_outer = np.mean(all_mae_outer)
    avg_avg = np.mean(all_mae_avg)
    overall = (avg_inner + avg_outer + avg_avg) / 3

    print("\n==============================")
    print("FINAL TEST RESULTS (MAE)")
    print("------------------------------")
    print(f"Inner: {avg_inner:.4f}C | Outer: {avg_outer:.4f}C | Avg: {avg_avg:.4f}C")
    print(f"OVERALL MAE: {overall:.4f}C")
    print("==============================")

if __name__ == "__main__":
    m, s = train_pinn()
    test_pinn(m, s)
