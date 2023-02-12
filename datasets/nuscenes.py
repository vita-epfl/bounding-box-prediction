import torch
import pandas as pd
from ast import literal_eval
import os
from PIL import Image
import time
import utils

class NuScenes(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir,
            out_dir,
            dtype,
            input,
            output,
            stride,
            skip=1,
            task='3D_bounding_box',
            from_file=True, # always set to True
            save=True
            ):
        
        print('*'*30)
        print('Loading NuScenes ', dtype, 'data (always from file)...')

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.input = input
        self.output = output
        self.stride = stride
        self.skip = skip
        self.dtype = dtype
        self.task = task
        self.save = save

        self.filename = 'nu_{}_{}_{}_{}.csv'.format(dtype, str(input),\
                                str(output), str(stride)) 
        
        sequence_centric = pd.read_csv(os.path.join(self.out_dir, self.filename))
        df = sequence_centric.copy()      
        for v in list(df.columns.values):
            print(v + ' loaded')
            try:
                df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]

        if self.save:
            sequence_centric.to_csv(os.path.join(self.out_dir, self.filename), index=False)

        self.data = sequence_centric.copy().reset_index(drop=True)
            
        print(sequence_centric.shape)
        print(self.dtype, "set loaded")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.input,self.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.output, self.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))

        if 'attribute' in self.task:
            true_attrib = torch.tensor([seq.label[i] for i in range(0,self.output, self.skip)])
            outputs.append(true_attrib.type(torch.float32))

        return tuple(outputs)