import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from engine.solver import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
import matplotlib.pyplot as plt
import os
import pandas as pd
from gluonts.dataset.pandas import PandasDataset

class SolarDataset(Dataset):
    def __init__(self, data, regular=True, pred_length=72):
        super(SolarDataset, self).__init__()
        self.sample_num = data.shape[0]
        self.samples = data
        self.regular = regular
        self.mask = np.ones_like(data)
        self.mask[:, -pred_length:, :] = 0.
        self.mask = self.mask.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.regular:
            return torch.from_numpy(x).float()
        mask = self.mask[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)

    def __len__(self):
        return self.sample_num
    
class Args_Example:
    def __init__(self) -> None:
        self.config_path = 'Config/etth_update.yaml'
        self.save_dir = 'forecasting_stocks'
        self.gpu = 0
        os.makedirs(self.save_dir, exist_ok=True)

# 处理gluonts 需要的pandas 时间格式
data_path="/git/datasets/ETTh.csv"
df_wide = pd.read_csv(data_path)
date_column='date'
df_wide[date_column]=pd.to_datetime(df_wide[date_column])
df_wide.set_index(date_column, inplace=True)
df_wide=df_wide.resample('1H').sum()
df_wide = df_wide[~df_wide.index.duplicated(keep='first')]
ds = PandasDataset(df_wide,target="OT")
data_grouper = MultivariateGrouper(max_target_dim=5)
data = data_grouper(ds)
data = data[0]['target'].transpose(1,0)
print(data.shape)





split_ratio=0.8
dim=36
split_num= int(data.shape[0]*split_ratio/dim)*dim
train = data[:split_num, :]
test = data[split_num+100:,:].reshape(1, -1, data.shape[-1])

scaler = MinMaxScaler()
train_scaled = normalize_to_neg_one_to_one(scaler.fit_transform(train))
test_scaled = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
test_scaled= normalize_to_neg_one_to_one(test_scaled)

train_dataset = SolarDataset(train_scaled.reshape(36, -1, data.shape[-1]))
dataloader = DataLoader(train_dataset, batch_size=18, shuffle=True, num_workers=0, drop_last=False, pin_memory=True, sampler=None)



args =  Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader':dataloader})

trainer.train()

_, seq_length, feat_num = test_scaled.shape
pred_length = 24

test_dataset = SolarDataset(test_scaled, regular=False, pred_length=pred_length)
real = scaler.inverse_transform(unnormalize_to_zero_to_one(test_scaled.reshape(-1, feat_num))).reshape(test_scaled.shape)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_scaled.shape[0], shuffle=False, num_workers=0, pin_memory=True, sampler=None)

sample, *_ = trainer.restore(test_dataloader, shape=[seq_length, feat_num], coef=1e-1, stepsize=5e-2, sampling_steps=200)
sample = scaler.inverse_transform(unnormalize_to_zero_to_one(sample.reshape(-1, feat_num))).reshape(test_scaled.shape)
mask = test_dataset.mask
mse = mean_squared_error(sample[~mask], real[~mask])
mape = mean_absolute_percentage_error(sample[~mask], real[~mask])
print(sample)
print(mse,mape)


plt.rcParams["font.size"] = 12

for idx in range(5):
    plt.figure(figsize=(15, 3))
    plt.plot(range(0, seq_length-pred_length), real[0, :(seq_length-pred_length), -idx], color='c', linestyle='solid', label='History')
    plt.plot(range(seq_length-pred_length-1, seq_length), real[0, -pred_length-1:, -idx], color='g', linestyle='solid', label='Ground Truth')
    plt.plot(range(seq_length-pred_length-1, seq_length), sample[0, -pred_length-1:, -idx], color='r', linestyle='solid', label='Prediction')
    plt.tick_params('both', labelsize=15)
    plt.subplots_adjust(bottom=0.1, left=0.05, right=0.99, top=0.95)
    plt.legend()
    plt.show()
    plt.savefig("results/forecaset_results.jpg")