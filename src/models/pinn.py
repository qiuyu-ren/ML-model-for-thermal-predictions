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

# === PATH LOGIC ===
# This detects the root folder (ML-model-for-thermal-prediction...) 
# regardless of whether you run it locally or on Colab.
try:
    # Path to this script (src/models/script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
except NameError:
    # Fallback for Colab interactive mode
    PROJECT_ROOT = os.getcwd()

# === 1. Physics Informed Loss Module ===
class PhysicsLoss(nn.Module):
    def __init__(self, scaler_min, scaler_scale):
        super(PhysicsLoss, self).__init__()
        self.k1 = nn.Parameter(torch.tensor([-1.0])) # Convection: Fluid to Inner
        self.k2 = nn.Parameter(torch.tensor([-1.0])) # Conduction: Avg to Outer
        self.k3 = nn.Parameter(torch.tensor([-1.0])) # Conduction: Inner to Avg

        self.register_buffer("s_min", torch.tensor(scaler_min, dtype=torch.float32))
        self.register_buffer("s_scale", torch.tensor(scaler_scale, dtype=torch.float32))

    def unscale(self, Ti_s, To_s, Ta_s, Text_s):
        # Column Indices: 1=Outer, 2=Inner, 3=Avg, 4=Input
        Ti = Ti_s * self.s_scale[2] + self.s_min[2]
        To = To_s * self.s_scale[1] + self.s_min[1]
        Ta = Ta_s * self.s_scale[3] + self.s_min[3]
        Text = Text_s * self.s_scale[4] + self.s_min[4]
        return Ti, To, Ta, Text

    def forward(self, preds, input_temps_scaled, dt=1.0):
        # preds contains [Ti, To, Ta]
        Ti, To, Ta, Text = self.unscale(preds[:, :, 0], preds[:, :, 1], preds[:, :, 2], input_temps_scaled)

        dTi_dt = (Ti[:, 1:] - Ti[:, :-1]) / dt
        dTo_dt = (To[:, 1:] - To[:, :-1]) / dt
        dTa_dt = (Ta[:, 1:] - Ta[:, :-1]) / dt

        Ti_c, To_c, Ta_c, Text_c = Ti[:, 1:], To[:, 1:], Ta[:, 1:], Text[:, 1:]
        k1, k2, k3 = torch.exp(self.k1), torch.exp(self.k2), torch.exp(self.k3)

        # Fluid -> Inner -> Avg -> Outer
        res_inner = dTi_dt - (k1 * (Text_c - Ti_c) - k3 * (Ti_c - Ta_c))
        res_avg   = dTa_dt - (k3 * (Ti_c - Ta_c) - k2 * (Ta_c - To_c))
        res_outer = dTo_dt - (k2 * (Ta_c - To_c))

        loss_ode = torch.mean(res_outer**2 + res_avg**2 + res_inner**2) / 100.0
        
        # Physical constraint: Ta should be between Ti and To
        lower = torch.min(Ti_c, To_c)
        upper = torch.max(Ti_c, To_c)
        penalty = torch.mean(torch.relu(lower - Ta_c) + torch.relu(Ta_c - upper))

        return loss_ode + (5.0 * penalty)

# === 2. Dataset ===
class ThermalWindowDataset(Dataset):
    def __init__(self, csv_paths, window_size=50):
        self.X, self.Y = [], []
        all_dfs = []
        for p in csv_paths:
            df = pd.read_csv(p).rename(columns={"T_ave (C)": "T_avg (C)"})
            all_dfs.append(df)

        combined = pd.concat(all_dfs)
        cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        self.scaler = MinMaxScaler().fit(combined[cols])

        for df in all_dfs:
            data = self.scaler.transform(df[cols])
            for i in range(0, len(data) - window_size, 5):
                # X: [Time, Outer, Inner, Avg, Input]
                self.X.append(data[i : i + window_size, [0, 1, 2, 3, 4]])
                # Y: [Inner, Outer, Avg]
                self.Y.append(data[i + 1 : i + window_size + 1, [2, 1, 3]])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# === 3. GRU Model ===
class ThermalGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=3, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.fc(out), h

# === 4. Training Loop ===
def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Updated paths using PROJECT_ROOT
    train_dir = os.path.join(PROJECT_ROOT, "data", "train", "data_in_150s")
    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    
    if not train_paths:
        raise FileNotFoundError(f"No CSV files found in {train_dir}. Check your folder structure.")

    dataset = ThermalWindowDataset(train_paths, window_size=50)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ThermalGRU(input_size=5).to(device)
    phys_fn = PhysicsLoss(dataset.scaler.min_, dataset.scaler.scale_).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': phys_fn.parameters(), 'lr': 0.01}
    ])

    print(f"Starting Training on {device}...")
    for epoch in range(100):
        model.train()
        total_d, total_p = 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            batch_preds, h = [], None
            curr_input = bx[:, 0:1, :]

            for t in range(bx.size(1)):
                out, h = model(curr_input, h)
                batch_preds.append(out)
                if t < bx.size(1) - 1:
                    if random.random() < 0.3:
                        # Reconstruct next input using predictions
                        curr_input = torch.cat([bx[:, t+1:t+2, 0:1], out[:, :, 1:2], out[:, :, 0:1], out[:, :, 2:3], bx[:, t+1:t+2, 4:5]], dim=-1)
                    else:
                        curr_input = bx[:, t+1:t+2, :]

            preds = torch.cat(batch_preds, dim=1)
            loss_data = torch.mean((preds - by)**2)
            loss_phys = phys_fn(preds, bx[:, :, 4])
            (loss_data + 0.5 * loss_phys).backward()
            optimizer.step()
            total_d += loss_data.item()
            total_p += loss_phys.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Data Loss: {total_d/len(train_loader):.6f} | Physics Loss: {total_p/len(train_loader):.6f}")

    return model, dataset.scaler

# === 5. Testing and Evaluation ===
def test_pinn(model, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_dir = os.path.join(PROJECT_ROOT, "data", "test", "test_in_150s")
    test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))
    output_dir = os.path.join(PROJECT_ROOT, "results", "PINN_Final")
    os.makedirs(output_dir, exist_ok=True)

    cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
    m_in, m_out, m_avg = [], [], []

    for path in test_paths:
        df = pd.read_csv(path).rename(columns={"T_ave (C)": "T_avg (C)"})
        scaled_data = scaler.transform(df[cols])
        preds, h = [], None
        curr_in = torch.tensor(scaled_data[0, :], dtype=torch.float32).view(1, 1, 5).to(device)

        with torch.no_grad():
            for t in range(len(scaled_data) - 1):
                out, h = model(curr_in, h)
                p = out.cpu().squeeze().numpy()
                preds.append(p)
                curr_in = torch.tensor([scaled_data[t+1, 0], p[1], p[0], p[2], scaled_data[t+1, 4]], dtype=torch.float32).view(1, 1, 5).to(device)

        # Unscaling logic
        def to_celsius(d_in, is_pred=False):
            d = np.zeros((len(d_in), 5))
            if is_pred: d[:, [2, 1, 3]] = d_in
            else: d[:, [2, 1, 3]] = d_in[:, [2, 1, 3]]
            return scaler.inverse_transform(d)

        inv_p = to_celsius(np.array(preds), True)
        inv_a = to_celsius(scaled_data[1:], False)

        m_in.append(np.mean(np.abs(inv_p[:, 2] - inv_a[:, 2])))
        m_out.append(np.mean(np.abs(inv_p[:, 1] - inv_a[:, 1])))
        m_avg.append(np.mean(np.abs(inv_p[:, 3] - inv_a[:, 3])))

    print("\n" + "="*30 + "\nFINAL TEST RESULTS (MAE)\n" + "-"*30)
    print(f"Inner: {np.mean(m_in):.4f}C | Outer: {np.mean(m_out):.4f}C | Avg: {np.mean(m_avg):.4f}C")
    print(f"OVERALL MAE: {(np.mean(m_in)+np.mean(m_out)+np.mean(m_avg))/3:.4f}C\n" + "="*30)

if __name__ == "__main__":
    m, s = train_pinn()
    test_pinn(m, s)
