import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import pandas as pd

from models import Multitask_PS_LSTM
from datasets import observationList

class args():
    def __init__(self):
        self.batch_size = 20
        self.scene_shape = [3, 240, 426]
        self.base_shape = [2048, 8, 14]
        self.hidden_size = 256
        self.sample = True
        self.n_train_sequences = 12000
        self.n_val_sequences = 3000
        self.trainOrVal = 'train'
        self.jaad_dataset = '../../../../data/smailait-data/jaad/annotations'
        self.dtype = 'train'
        self.from_file = True
        self.file = 'input_data.csv'
        self.seq_len = 18
        self.loader_workers = 8
        self.loader_shuffle = True
        self.pin_memory = False
        self.image_resize = [240, 426]
        self.image_size = [1080, 1920]
        self.device='cuda'
        self.predict_intention = True
        self.use_scene_features = False
        self.predict_speed = True
        self.log_file = 'log_Multitask_PS_LSTM.txt'
        self.lr = 0.0001
        self.epochs = 100
        self.network_save_path = 'Multitask_PS_LSTM_100epochs.pkl'
        args.output_file = 'output.csv'


def main():
    args = args()
    net = Multitask_PS_LSTM(args).to(args.device)
    data = observationList(args)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    position_outputs = []
    intention_outputs = []

    for i, (obs_speed, obs_pos) in enumerate(data_loader):
        obs_speed = obs_speed.to(args.device)
        obs_pos = obs_pos.to(args.device)

        with torch.no_grad():
            speed_preds, intention_preds = model(speed=obs_speed, pos=obs_pos)
            preds_p = utils.speed2pos(speed_preds, obs_pos, args.batch_size, args.device)

        position_outputs.append(preds_p.tolist())
        intention_outputs.append(intention_outputs.tolist())

    predicted_pos = pd.DataFrame(position_outputs)
    predicted_intent = pd.DataFrame(intention_outputs)

    predictions = predicted_pos.join(predicted_intent)

    predictions.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()
