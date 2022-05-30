import torch
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image
import time
import utils


class myJTA(torch.utils.data.Dataset):
    def __init__(self, args, occluded=True, normalize=False):
        print('Loading', args.dtype, 'data ...')
        
        if(args.from_file):
            sequence_centric = pd.read_csv(args.save_path)
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
                
            if args.save:
                sequence_centric.to_csv(args.save_path, index=False)
            
            self.data = sequence_centric.copy().reset_index(drop=True)
            
        else: #read data 
            print('Processing data ...')
            sequence_centric = pd.DataFrame()
            for file in glob.glob(os.path.join(args.dataset, args.dtype,"*")):
                df = pd.read_csv(file)
                if not df.empty:
                    print(file)
                    
                    df = df.reset_index(drop = True) #reset index
                    df['bounding_box'] = df[['x', 'y', 'z', 'w', 'h', 'd']].apply(lambda row: \
                                    [round(row.x,8), round(row.y,8), round(row.z,8),\
                                     round(row.w,8), round(row.h,8), round(row.d,8)], axis=1)
                    
                    bb = df.groupby(['ID'])['bounding_box'].apply(list).reset_index(name='bounding_box')
                    s = df.groupby(['ID'])['scenefolderpath'].apply(list).reset_index(name='scenefolderpath').drop(columns='ID')
                    f = df.groupby(['ID'])['frame'].apply(list).reset_index(name='frame').drop(columns='ID')
                    m = df.groupby(['ID'])['mask'].apply(list).reset_index(name='mask').drop(columns='ID')
                    d = bb.join(s).join(f).join(m)
                    
                    d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < args.input + args.output)].index) # drops values with not enough frames
                    d = d.reset_index(drop=True)
                    
                    SKIP = args.skip
                    INPUT = args.input
                    OUTPUT = args.output
                    STRIDE = args.stride
                    
                    bounding_box_o = np.empty((0,INPUT,6))
                    bounding_box_t = np.empty((0,OUTPUT,6))
                    scene_o = np.empty((0,1))
                    file = np.empty((0,INPUT))   
                    mask = np.empty((0,INPUT))
                    ind = np.empty((0,1))
        
                    for i in range(d.shape[0]):
                        ped = d.loc[i]
                        k = 0
                        END = k + (INPUT + OUTPUT)*SKIP
                        
                        while (END) <= len(ped.bounding_box):
                            START = k
                            MID = k + INPUT*SKIP
                            
                            obs_frames = ped.frame[START:MID:SKIP]
                            pred_frames = ped.frame[MID:END:SKIP]
                
                            # check if frames are continuous
                            if utils.check_continuity(obs_frames, SKIP) or utils.check_continuity(pred_frames, SKIP):
                                pass
                            
                            if not occluded:
                                num_obs_masked = np.sum(ped['mask'][START:MID:SKIP])
                                num_pred_masked = np.sum(ped['mask'][MID:END:SKIP])
                            
                                # check if ped is unmasked in all frames
                                if num_obs_masked != 0 or num_pred_masked != 0:
                                    pass
                            
                            # get sequences of obs and preds
                            ind = np.vstack((ind, ped['ID']))
                            
                            if normalize:
                                whole_seq = np.array(ped.bounding_box[START:END])
                                baseline = np.array(whole_seq[0][0:3])
                                for i in range(len(whole_seq)):
                                    whole_seq[i][0:3] = np.round((whole_seq[i][0:3] - baseline), 4)

                                bounding_box_o = np.vstack((bounding_box_o, whole_seq[0:INPUT*SKIP:SKIP].reshape(1,INPUT,6)))
                                bounding_box_t = np.vstack((bounding_box_t, whole_seq[INPUT:(INPUT+OUTPUT)*SKIP:SKIP].reshape(1,OUTPUT,6)))  
                            else:
                                bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[START:MID:SKIP]).reshape(1,INPUT,6)))
                                bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[MID:END:SKIP]).reshape(1,OUTPUT,6)))  
                            
                            scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k]).reshape(1,1)))
                            filename = ['%d'%int(x) +'.jpg' for x in obs_frames]
                            file = np.vstack((file, np.array(filename)))    
                            mask_values = np.array(ped['mask'][START:MID:SKIP]).astype(int).reshape(1,INPUT)
                            mask = np.vstack((mask, mask_values))

                            k += STRIDE
                    
                    dt = pd.DataFrame({'ID':ind.reshape(-1)})
                    data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, INPUT, 6).tolist(),
                                         'future_bounding_box':bounding_box_t.reshape(-1, 1, OUTPUT, 6).tolist(),
                                         'scenefolderpath':scene_o.reshape(-1,1).tolist(),
                                         'filename':file.reshape(-1,INPUT).tolist(),
                                         'mask': mask.reshape(-1,INPUT).tolist()
                                         })
                    data.bounding_box = data.bounding_box.apply(lambda x: x[0])
                    data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
                    data = dt.join(data)
                    
                    sequence_centric = sequence_centric.append(data, ignore_index=True)
            
        if args.save:
            sequence_centric.to_csv(args.save_path, index=False)
            
        self.data = sequence_centric.copy().reset_index(drop=True)
            
        self.args = args
        self.dtype = args.dtype
        print(sequence_centric.shape)
        print(args.dtype, "set loaded")
        print('*'*30)
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.args.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))

        return tuple(outputs)
    

class myNuscenes(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        
        sequence_centric = pd.read_csv(args.save_path)
        df = sequence_centric.copy()      
        for v in list(df.columns.values):
            print(v+' loaded')
            try:
                df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]

        if args.save:
            sequence_centric.to_csv(args.save_path, index=False)

        self.data = sequence_centric.copy().reset_index(drop=True)
            
        self.args = args
        self.dtype = args.dtype
        print(sequence_centric.shape)
        print(args.dtype, "set loaded")
        print('*'*30)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.args.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))

        return tuple(outputs)

    
def data_loader(args, data):
    if data == 'nuscenes':
        dataset = myNuscenes(args)
    if data == 'JTA':
        dataset = myJTA(args)
        
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return dataloader


class args():
    def __init__(self, dtype):
        self.dataset = '/home/yju/JTA/preprocessed_annotations' #folder containing parsed annotations (used when first time loading data)
        self.dtype        = dtype
        self.from_file    = False #read dataset from csv file or reprocess data
        self.save         = True
        self.output_path   = '/home/yju/JTA'
        self.model_name    = '3d_bbox_trained.pkl'
        self.loader_workers = 12
        self.loader_shuffle = True
        self.pin_memory     = False
        self.device         = 'cuda'
        self.batch_size     = 128 # 32, 64
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.input  = 30
        self.output = 30
        self.stride = 60
        self.skip   = 2
        self.task   = 'bounding_box'  
        self.lr = 0.001 
        self.save_subset = False
        self.subset = 1000
        self.filename     = 'jta_{}_{}_{}_{}.csv'.format(str(self.dtype), str(self.input),\
                            str(self.output), str(self.stride)) 
        self.save_path = os.path.join(self.output_path, self.filename)
        
# class args():
#     def __init__(self, dtype):
#         self.dataset = '/work/vita/datasets/NuScenes_full/US'
#         self.dtype        = dtype
#         self.from_file    = True #read dataset from csv file or reprocess data
#         self.save         = True
#         self.output_path   = '/home/yju/NuScenes'
#         self.model_name    = 'model_0.001_512_scheduler.pkl'
#         self.loader_workers = 12
#         self.loader_shuffle = True
#         self.pin_memory     = False
#         self.device         = 'cuda'
#         self.batch_size     = 128 # 32, 64
#         self.n_epochs       = 100
#         self.hidden_size    = 512
#         self.hardtanh_limit = 100
#         self.input  = 4
#         self.output = 4
#         self.stride = 8
#         self.skip   = 1
#         self.task   = 'bounding_box'
#         self.use_scenes = False      
#         self.lr = 0.00001 
#         self.save_subset = False
#         self.subset = 1000
#         self.filename     = 'nu_{}_{}_{}_{}.csv'.format(str(self.dtype), str(self.input),\
#                             str(self.output), str(self.stride)) 
#         self.save_path = os.path.join(self.output_path, self.filename)
        

if __name__ == '__main__':
    train = data_loader(args(dtype='train'))  
#     val = data_loader(args('val')) 
    test = data_loader(args('test')) 
