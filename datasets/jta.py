import torch
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image
import time
import utils


    
class JTA(torch.utils.data.Dataset):
    def __init__(self, 
                data_dir,
                out_dir,
                dtype,
                input,
                output,
                stride,
                skip=1,
                task='bounding_box',
                from_file=False,
                save=True,
                occluded=True, 
                normalize=False
                ):
        
        print('*'*30)
        print('Loading JTA', dtype, 'data ...')

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.input = input
        self.output = output
        self.stride = stride
        self.skip = skip
        self.dtype = dtype
        self.task = task

        self.filename = 'jta_{}_{}_{}_{}.csv'.format(dtype, str(input),\
                                str(output), str(stride)) 

        if(from_file):
            sequence_centric = pd.read_csv(os.path.join(self.out_dir, self.filename))
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
                
            if save:
                sequence_centric.to_csv(os.path.join(self.out_dir, self.filename), index=False)
            
            self.data = sequence_centric.copy().reset_index(drop=True)
            
        else: #read data 
            print('Reading data files ...')
            sequence_centric = pd.DataFrame()
            print('Processing data ...')
            for file in glob.glob(os.path.join(self.data_dir, dtype,"*")):
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
                    
                    d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < input + output)].index) # drops values with not enough frames
                    d = d.reset_index(drop=True)
                    
                    bounding_box_o = np.empty((0, input, 6))
                    bounding_box_t = np.empty((0, output, 6))
                    scene_o = np.empty((0,1))
                    file = np.empty((0, input))   
                    mask = np.empty((0, input))
                    ind = np.empty((0,1))
        
                    for i in range(d.shape[0]):
                        
                        ped = d.loc[i]
                        # print(len(ped.bounding_box))
                        k = 0
                        
                        while (k + (input + output) * skip) <= len(ped.bounding_box):
    
                            START = k
                            MID = k + input*skip
                            END = k + (input + output) * skip
                            
                            obs_frames = ped.frame[START:MID:skip]
                            pred_frames = ped.frame[MID:END:skip]

                            # check if frames are continuous
                            if utils.check_continuity(obs_frames, skip) or utils.check_continuity(pred_frames, skip):
                                pass
                            
                            if not occluded:
                                # print('check occlusion')
                                num_obs_masked = np.sum(ped['mask'][START:MID:skip])
                                num_pred_masked = np.sum(ped['mask'][MID:END:skip])
                            
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

                                bounding_box_o = np.vstack((bounding_box_o, whole_seq[0:input*skip:skip].reshape(1,input,6)))
                                bounding_box_t = np.vstack((bounding_box_t, whole_seq[input:(input+output)*skip:skip].reshape(1,output,6)))  
                            
                            else:
                                bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[START:MID:skip]).reshape(1,input,6)))
                                bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[MID:END:skip]).reshape(1,output,6)))  
                            
                            scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k]).reshape(1,1)))
                            filename = ['%d'%int(x) +'.jpg' for x in obs_frames]
                            file = np.vstack((file, np.array(filename)))    
                            mask_values = np.array(ped['mask'][START:MID:skip]).astype(int).reshape(1,input)
                            mask = np.vstack((mask, mask_values))

                            k += stride
                    
                    dt = pd.DataFrame({'ID':ind.reshape(-1)})
                    data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, input, 6).tolist(),
                                         'future_bounding_box':bounding_box_t.reshape(-1, 1, output, 6).tolist(),
                                         'scenefolderpath':scene_o.reshape(-1,1).tolist(),
                                         'filename':file.reshape(-1,input).tolist(),
                                         'mask': mask.reshape(-1,input).tolist()
                                         })
                    data.bounding_box = data.bounding_box.apply(lambda x: x[0])
                    data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
                    data = dt.join(data)
                    
                    sequence_centric = sequence_centric.append(data, ignore_index=True)
            
        if save:
            sequence_centric.to_csv(os.path.join(self.out_dir, self.filename), index=False)
            
        self.data = sequence_centric.copy().reset_index(drop=True)

        print(sequence_centric.shape)
        print(dtype, "set loaded")
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.input,self.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.output,self.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))

        return tuple(outputs)