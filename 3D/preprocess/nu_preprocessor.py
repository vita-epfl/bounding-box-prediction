# -*- coding: utf-8 -*-
"""
Author: Celinna Ju
Purpose: Preprocessing NuScenes dataset for 3D bounding box prediction.
    This scripts gets the x,y,z,w,l,h information (3D bbox) for each 
    pedestrian in each frame of a video

"""
import os
import sys
import time
import math
import copy
import json
import logging
from collections import defaultdict
import datetime
import pandas as pd

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


class PreprocessNuscenes:
    """Preprocess Nuscenes dataset"""
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1

    CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
#     CAMERAS = ('CAM_FRONT', )
    print("!Only frontal camera!")
    
    data = {'train': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = []),
              'val': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = []),
              'test': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = [])
              }

    def __init__(self, dir_nuscenes):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not os.path.isdir(PREPROCESSED_DATA_DIR):    
            os.mkdir(PREPROCESSED_DATA_DIR)      
        
#         self.input = args.input
#         self.output = args.output
        self.input = 4
        self.output = 4
        self.total = self.input + self.output
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.nusc, self.scenes, self.split_train, self.split_val, self.split_test = factory(dir_nuscenes)

    def run(self):
        """
        Prepare arrays 
        """
        cnt_obs = cnt_samples = cnt_ann = 0
        start = time.time()    
        
        for i in range(len(self.nusc.instance)):
            cam_data = defaultdict(lambda: defaultdict(list))
            my_instance = self.nusc.instance[i]
            current_cat = self.nusc.get('category', my_instance['category_token'])

            if current_cat['name'][:6] != 'animal':
                general_name = current_cat['name'].split('.')[0] + '.' + current_cat['name'].split('.')[1]
            else:
                general_name = 'animal'
            
            if general_name in select_categories('person'):
                first_token = my_instance['first_annotation_token']    
                last_token = my_instance['last_annotation_token']
                nbr_samples = my_instance['nbr_annotations']
                current_token = first_token
                
                while current_token != last_token:
                    # get current annotations
                    current_ann = self.nusc.get('sample_annotation', current_token)
                    current_sample = self.nusc.get('sample', current_ann['sample_token'])
                    scene = self.nusc.get('scene', current_sample['scene_token'])
                    
                    # check scene belongs to which train/val/test
                    if scene['name'] in self.split_train:
                        phase = 'train'
                    elif scene['name'] in self.split_val:
                        phase = 'val'
                    elif scene['name'] in self.split_test:
                        phase = 'test'
                    else:
                        print("phase name not in training or validation split")
                        continue
                    
                    # check for each camera
                    for cam in self.CAMERAS:
                        cam_token = current_sample['data'][cam]  # sample data token
                        ann_token = current_ann['token'] # annotation token
                        path_im, box_obj, kk = self.nusc.get_sample_data(cam_token, box_vis_level=1, selected_anntokens=[ann_token])
                        
                        # Obtain 3D box in camera coords
                        # each box obj should only have either 0 or 1 unique box value
                        if not box_obj:
                            pass
                        
                        else:
                            cam_data[cam]['cam_token'].append(cam_token)
                            #can extract location using 
                            #nusc.get('sample_data', cam_token)
                            cam_data[cam]['ann_token'].append(ann_token)
                            cam_data[cam]['idx'].append(cnt_ann) # tracks if frames are continuous

                            box_obj = box_obj[0] 
                            whd = [float(box_obj.wlh[i]) for i in (0, 2, 1)]
                            bbox_3d = box_obj.center.tolist() + whd
                            cam_data[cam]['bbox'].append(bbox_3d)
                            cnt_samples += 1
                            
                        # check if there are sufficient data to save
                        if (len(cam_data[cam]['bbox']) == self.total) and not check_continuity(cam_data[cam]['idx'], 1):
                            self.data[phase]['cam_token'].append(cam_data[cam]['cam_token'])
                            self.data[phase]['ann_token'].append(cam_data[cam]['ann_token'])
                            self.data[phase]['bounding_box'].append(cam_data[cam]['bbox'][0:self.input])
                            self.data[phase]['future_bounding_box'].append(cam_data[cam]['bbox'][self.input:self.total])
                            self.data[phase]['K'].append(kk)
                            
                            # reset values
                            cam_data[cam]['bbox'] = []
                            cam_data[cam]['cam_token'] = []
                            cam_data[cam]['ann_token'] = []
                            cam_data[cam]['exists'] = []
                            
                            cnt_obs += 1

                    next_token = current_ann['next']
                    current_token = next_token
                    cnt_ann += 1
                    
        df_train = pd.DataFrame.from_dict(self.data['train'])
        df_val = pd.DataFrame.from_dict(self.data['val'])
        df_test = pd.DataFrame.from_dict(self.data['test'])
        
        # save files
        location = '/home/yju/NuScenes/'
        file1 = 'train_4_4_1_1.csv'
        file2 = 'val_4_4_1_1.csv'
        file3 = 'test_4_4_1_1.csv'
        
        df_train.to_csv(os.path.join(location, file1), index=False)
        df_val.to_csv(os.path.join(location, file2), index=False)
        df_test.to_csv(os.path.join(location, file3), index=False)
        end = time.time() 
        
        print("\nSaved {} observations for {} samples of {} annotations. Total time: {:.1f} minutes"
              .format(cnt_obs, cnt_samples, cnt_ann, (end-start)/60))
        print("\nOutput files: \n{}\n{}\n{}".format(file1, file2, file3))
            
            
def factory(dir_nuscenes):
    """Define dataset type and split training and validation"""

    version = 'v1.0-trainval'

    nusc = NuScenes(version=version, dataroot=dir_nuscenes, verbose=True)
    scenes = nusc.scene

    split_scenes = splits.create_splits_scenes()
    split_train, split_val, split_test = split_scenes['train'], split_scenes['val'], split_scenes['test']

    return nusc, scenes, split_train, split_val, split_test

def select_categories(cat):
    """
    Choose the categories to extract annotations from
    """
    assert cat in ['person', 'all', 'car', 'cyclist']

    if cat == 'person':
        categories = ['human.pedestrian']
    elif cat == 'all':
        categories = ['human.pedestrian', 'vehicle.bicycle', 'vehicle.motorcycle']
    elif cat == 'cyclist':
        categories = ['vehicle.bicycle']
    elif cat == 'car':
        categories = ['vehicle']
    return categories


def check_continuity(my_list, skip):
    '''
    Checks if there are frames continuously
    Returns True is there is discontinuity
    '''
    return any(a+skip != b for a, b in zip(my_list, my_list[1:]))

        
if __name__ == '__main__':
    path = '/work/vita/datasets/nuScenes-prediction/v1.0-full'
    
#     raw_data = ['test', 'val']
    raw_data = ['train']
               
    for data_type in raw_data:
        print('Reading {} data...'.format(data_type))
        preprocessor = JTAPreprocessor(dataset_path=path)
        preprocessor.normal(data_type=data_type)
