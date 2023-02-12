# -*- coding: utf-8 -*-
"""
Author: Celinna Ju
Purpose: Preprocessing JTA dataset for 3D bounding box prediction.
    This scripts gets the x,y,z,w,l,h information (3D bbox) for each 
    pedestrian in each frame of a video

"""
import json
import os
import re
import pandas as pd
import numpy as np
import argparse

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='JTA preprocessing')
    
    parser.add_argument('--data_dir', type=str,
                        help='Path to dataset',
                        required=True)

    args = parser.parse_args()

    return args


class JTAPreprocessor():
    dataset_total_frame_num = 900
    
    def __init__(self, data_dir, custom_name=None):
        self.data_dir = data_dir
        self.custom_name = custom_name

        self.start_dim = 5
        self.end_dim = 8
        self.occluded_idx = 8

        # Create folder in original dataset
        self.out_dir = os.path.join(self.data_dir, 'processed_annotations')
        if not os.path.isdir(self.out_dir):    
            os.mkdir(self.out_dir)

    
    def bbox3d_padded(self, x_min, y_min, z_min, x_max, y_max, z_max, \
                      h_inc_perc=0.15, w_inc_perc=0.1, d_inc_perc=0.1):
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min

        inc_h = (height * h_inc_perc) / 2
        inc_w = (width * w_inc_perc) / 2
        inc_d = (depth * d_inc_perc) / 2

        x_min = x_min - inc_w
        x_max = x_max + inc_w

        y_min = y_min - inc_h
        y_max = y_max + inc_h
        
        z_min = z_min - inc_d
        z_max = z_max + inc_d

        width = np.round((x_max - x_min),4)
        height = np.round((y_max - y_min),4)
        depth = np.round((z_max - z_min),4)
        
        x = np.round(x_min + width/2, 4)
        y = np.round(y_min + height/2, 4)
        z = np.round(z_min + depth/2, 4)

       # return np.round(x_min,4), np.round(y_min,4), np.round(z_min,4), width, height, depth
        return x, y, z, width, height, depth
    
    def normal(self, data_type):
        print('='*100)
        print('Parsing {} data ...'.format(data_type))
        
        data_dir = os.path.join(self.data_dir, 'annotations', data_type)
        
        if not os.path.isdir(os.path.join(self.out_dir, data_type)):    
            os.mkdir(os.path.join(self.out_dir, data_type))
        
        image_location =  os.path.join(self.data_dir, 'frames', data_type)
        
        # parsing all data
        for entry in os.scandir(data_dir):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                video = re.search("(seq_\d+).json", entry.name).group(1)
                
                if self.custom_name is not None:
                    filename = video + '_' + self.custom_name + '.csv'
                    final_path = os.path.join(self.out_dir, data_type, filename)
                else:
                    filename = video +'.csv'
                    final_path = os.path.join(self.out_dir, data_type, filename)
                    
                #assert os.path.exists(final_path) is False, f"preprocessed file exists at {path}"
                if os.path.exists(final_path):
                    print('File {} exists...skipped...'.format(entry.name))
                    continue

                # load original JTA json file
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                image_path = os.path.join(image_location, video)
                
                data = np.empty((0,9))
                
                # for each frame in video
                for i in range(1, self.dataset_total_frame_num+1):
                   # get specific frame
                    frame = matrix[matrix[:, 0] == i]  
                
                    # get data for each person in frame
                    peds = np.unique(frame[:,1]).astype(int)
                    N = len(peds)
                    
                    X_min = np.ones(N)*np.inf
                    Y_min = np.ones(N)*np.inf
                    Z_min = np.ones(N)*np.inf
                    X_max = np.ones(N)*-np.inf
                    Y_max = np.ones(N)*-np.inf
                    Z_max = np.ones(N)*-np.inf
                    
                    masks = np.zeros(N)
                    
                    for pose in frame:
                        pid = int(pose[1])
                        idx = np.argwhere(peds == pid).item()
                        
                        if pose[self.start_dim] < X_min[idx]:
                            X_min[idx] = pose[self.start_dim]
                        if pose[self.start_dim+1] < Y_min[idx]:
                            Y_min[idx] = pose[self.start_dim+1]
                        if pose[self.start_dim+2] < Z_min[idx]:
                            Z_min[idx] = pose[self.start_dim+2]
                        if pose[self.start_dim] > X_max[idx]:
                            X_max[idx] = pose[self.start_dim]
                        if pose[self.start_dim+1] > Y_max[idx]:
                            Y_max[idx] = pose[self.start_dim+1]
                        if pose[self.start_dim+2] > Z_max[idx]:
                            Z_max[idx] = pose[self.start_dim+2]
                            
                        # check if joints are not occluded (not including self-occlusion)
                        masks[idx] = pose[self.occluded_idx]
                                   
                    [x, y, z, width, height, depth] = self.bbox3d_padded(X_min, Y_min, Z_min, X_max, Y_max, Z_max)
            
                    ID = np.array(peds + 1).astype(int)
                
                    frame_num = np.ones(N)*i
                    frame_num = frame_num.astype(int)
                    
                    ped_data = np.array([frame_num, ID, x, y, z, width, height, depth, masks]).T
                    data = np.vstack((data,ped_data))
                    
                # write data for each file    
                data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                  'ID': data[:,1].reshape(-1), 
                                  'x': data[:,2].reshape(-1), 
                                  'y': data[:,3].reshape(-1), 
                                  'z': data[:,4].reshape(-1), 
                                  'w': data[:,5].reshape(-1), 
                                  'h': data[:,6].reshape(-1), 
                                  'd': data[:,7].reshape(-1), 
                                  'mask': data[:,8].reshape(-1)
                                  })
                
                data_to_write['scenefolderpath'] = image_path
                data_to_write['filename'] = data_to_write.frame
                data_to_write.filename = data_to_write.filename.apply(lambda x: '%d'%int(x)+'.jpg')
                data_to_write.to_csv(final_path, index=False)
                print('File {} saved... '.format(filename))
        
        print("Done!")        
        return data_to_write


        
if __name__ == '__main__':
    args = parse_args()

    raw_data = ['train', 'test', 'val']
               
    for data_type in raw_data:
        preprocessor = JTAPreprocessor(data_dir=args.data_dir)
        preprocessor.normal(data_type=data_type)
