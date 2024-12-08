from Model import Model
from Dataloader import TimeSeriesDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

csv_file_path = './csvfile.csv'
data = pd.read_csv(csv_file_path)

if '%time' in data.columns:
    data['%time'] = pd.to_datetime(data['%time'])

feature_columns = ['Tout', 'Rhout', 'Iglob', 'RadSum', 'Windsp', 'AbsHumOut',
                   'VentLee', 'Ventwind', 'AssimLight', 'EnScr', 'BlackScr',
                   'PipeGrow', 'PipeLow', 'co2_dos']

X = data[feature_columns].values

window_size = 6

X_seq = []
for i in range(len(X) - window_size):
    X_seq.append(X[i:i + window_size])
X_seq = torch.tensor(X_seq, dtype=torch.float32)

dataset = TimeSeriesDataset(X_seq, torch.zeros(len(X_seq), 1)) 
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  

args = {
    'cuda': False,
    'window': window_size,
    'hidRNN': 64,
    'hidCNN': 32,
    'hidSkip': 8,
    'CNN_kernel': 3,
    'skip': 2,
    'highway_window': 5,
    'dropout': 0.2,
    'model': 'attn',
    'attn_score': 'scaled_dot',
    'output_fun': None
}
data_args = {'m': X_seq.shape[2]}  
model = Model(args, data_args)


model_save_path = 'models\LSTMNETATT_model.pth' 
model.load_state_dict(torch.load(model_save_path))


model.eval()


predictions = []
with torch.no_grad():
    for X_batch, _ in dataloader:  
        y_pred = model(X_batch) 
        predictions.append(y_pred.numpy())  

predictions = np.concatenate(predictions, axis=0)

print("Predictions:")
print(predictions)
output_csv_path = '/content/predictions.csv'
pd.DataFrame(predictions, columns=[f"Pred_Target_{i+1}" for i in range(predictions.shape[1])]).to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")