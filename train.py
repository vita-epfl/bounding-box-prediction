import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from models import Multitask_PS_LSTM
from datasets import data_loader
from trainer import train, test


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
        self.file = '/data/smailait-data/train_crossing.csv'
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

def main():
    args = args()
    log = open(args.log_file, 'w+')

    net = Multitask_PS_LSTM(args).to(args.device)
    train, val, test = data_loader(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold = 1e-8, verbose=True)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    for epoch in range(args.epochs):
        train(net, args.device, train, optimizer, scheduler, epoch, mse, bce, log)
        test(net, args.device, val, epoch, mse, bce, log)

    test(net, args.device, test, epoch, mse, bce, log)
    torch.save(net.state_dict(), 'Multitask_PS_LSTM_100epochs.pkl')

    plt.figure(figsize=(10,8))
    plt.plot(list(range(len(train_s_scores))), train_s_scores, label = 'Training loss')
    plt.plot(list(range(len(val_s_scores))), val_s_scores, label = 'Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('Mean square error loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
