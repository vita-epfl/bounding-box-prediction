# -*- coding: utf-8 -*-
"""
Author: Celinna Ju
Purpose: Preprocessing NuScenes dataset for 3D bounding box prediction.
    This scripts gets the x,y,z,w,l,h information (3D bbox) for each 
    pedestrian in each frame of a video
"""
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )

import logging
from collections import defaultdict
import datetime
import pandas as pd
import argparse
import utils 

from nuscenes.nuscenes import NuScenes
from nu_split import create_splits_scenes

# from sklearn.utils import resample

attribute_mapping = {   "standing" : [1, 0, 0],
                        "moving" : [0, 1, 0],
                        "sitting_lying_down" : [0, 0, 1]
                    }

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Nuscenes preprocessor...')
    
    parser.add_argument('--data_dir', type=str, help='Path to dataset', required=True)
    parser.add_argument('--version', type=str, help='Dataset version', default='v1.0-mini')
    parser.add_argument('--input', type=int, help='Input size', default=4)
    parser.add_argument('--output', type=int, help='Output size', default=4)

    args = parser.parse_args()

    return args


class NuscenesPreprocessor():
    """Preprocess Nuscenes dataset"""
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1

    CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
#     CAMERAS = ('CAM_FRONT', )
    print("!Only frontal camera!")
    
    data = {'train': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = [], attribute = [], label = [], future_label=[]),
              'val': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = [], attribute = [], label = [], future_label=[]),
              'test': dict(cam_token=[], ann_token=[], bounding_box=[], future_bounding_box=[],
                            K = [], attribute = [], label = [], future_label=[])
              }

    def __init__(self, dataset_path, version, input, output):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
             
        self.dataset_path = dataset_path
        self.version = version
        self.input = input
        self.output = output
        self.total = self.input + self.output
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.nusc, self.scenes, self.split_train, self.split_val, self.split_test = factory(dataset_path, version)

        # Create folder in original dataset
        self.out_dir = os.path.join(dataset_path, f'nu_{self.input}_{self.output}_{self.total}')
        if not os.path.isdir(self.out_dir):    
            os.mkdir(self.out_dir)

        self.run()


    def run(self):
            """
            Prepare arrays 
            """
            cnt_obs = cnt_samples = cnt_ann = 0
            # start = time.time()    
            
            for i in range(len(self.nusc.instance)):
                cam_data = defaultdict(lambda: defaultdict(list))
                my_instance = self.nusc.instance[i]
                current_cat = self.nusc.get('category', my_instance['category_token'])

                if current_cat['name'][:6] != 'animal':
                    general_name = current_cat['name'].split('.')[1]
                else:
                    general_name = 'animal'
            
                if general_name == 'pedestrian':
                    first_token = my_instance['first_annotation_token']    
                    last_token = my_instance['last_annotation_token']
                    nbr_samples = my_instance['nbr_annotations']
                    current_token = first_token
                    
                    while current_token != last_token:
                        # get current annotations
                        current_ann = self.nusc.get('sample_annotation', current_token)
                        current_sample = self.nusc.get('sample', current_ann['sample_token'])
                        if len(current_ann['attribute_tokens']) == 0:
                            cam_data[cam]['bbox'] = []
                            cam_data[cam]['cam_token'] = []
                            cam_data[cam]['ann_token'] = []
                            cam_data[cam]['attribute'] = []
                            cam_data[cam]['label'] = []
                            pass
                        else:
                            current_attr = self.nusc.get('attribute', current_ann['attribute_tokens'][0])['name']
                            attribute = current_attr.split('.')[1]
                            if attribute not in attribute_mapping:
                                cam_data[cam]['bbox'] = []
                                cam_data[cam]['cam_token'] = []
                                cam_data[cam]['ann_token'] = []
                                cam_data[cam]['attribute'] = []
                                cam_data[cam]['label'] = []
                                pass
                            else:
                                one_hot_attr = attribute_mapping[attribute]
                                #print(general_name, attribute, one_hot_attr)
                        
                                scene = self.nusc.get('scene', current_sample['scene_token'])

                                # check scene belongs to which train/val/test
                                if scene['name'] in self.split_train:
                                    phase = 'train'
                                elif scene['name'] in self.split_val:
                                    phase = 'val'
                                elif scene['name'] in self.split_test:
                                    phase = 'test'
                                    print('test')
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
                                        cam_data[cam]['attribute'].append(attribute)
                                        cam_data[cam]['label'].append(one_hot_attr)

                                        box_obj = box_obj[0] 
                                        whd = [float(box_obj.wlh[i]) for i in (0, 2, 1)]
                                        bbox_3d = box_obj.center.tolist() + whd
                                        cam_data[cam]['bbox'].append(bbox_3d)
                                        cnt_samples += 1

                                    # check if there are sufficient data to save
                                    if (len(cam_data[cam]['bbox']) == self.total) and not utils.check_continuity(cam_data[cam]['idx'], 1):
                                        self.data[phase]['cam_token'].append(cam_data[cam]['cam_token'])
                                        self.data[phase]['ann_token'].append(cam_data[cam]['ann_token'])
                                        self.data[phase]['bounding_box'].append(cam_data[cam]['bbox'][0:self.input])
                                        self.data[phase]['future_bounding_box'].append(cam_data[cam]['bbox'][self.input:self.total])
                                        self.data[phase]['K'].append(kk)
                                        self.data[phase]['attribute'].append(cam_data[cam]['attribute'])
                                        self.data[phase]['label'].append(cam_data[cam]['label'][0:self.input])
                                        self.data[phase]['future_label'].append(cam_data[cam]['label'][self.input:self.total])

                                        # reset values
                                        cam_data[cam]['bbox'] = []
                                        cam_data[cam]['cam_token'] = []
                                        cam_data[cam]['ann_token'] = []
                                        cam_data[cam]['attribute'] = []
                                        cam_data[cam]['label'] = []

                                        cnt_obs += 1

                        next_token = current_ann['next']
                        current_token = next_token 
                        cnt_ann += 1
    

    def preprocess(self, data_type):
        df = pd.DataFrame.from_dict(self.data[data_type])

        filename = 'nu_{}_{}_{}_{}.csv'.format(data_type, self.input, self.output, self.total)

        # if not os.path.isdir(os.path.join(self.out_dir, data_type)):    
        #     os.mkdir(os.path.join(self.out_dir, data_type))
        
        # df['attrib_label'] = df['attribute'].map(attribute_mapping)
        
        # upsampling
        # df = df_train.copy()
        # attribs = []
        # class_a = []
        # for index, row in df.iterrows():
        #     attribs.extend(row['attribute'])
        #     class_a.append(row['attribute'][-1])
        # df['class'] = class_a

        # num_majority = df[df['class'] == 'moving']['class'].size
        # # Upsample minority class
        # df_moving = df[df['class']=='moving']
        # df_standing_upsampled = resample(df[df['class']=='standing'], 
        #                                 replace=True,     # sample with replacement
        #                                 n_samples=num_majority ,    # to match majority class
        #                                 random_state=42) # reproducible results
        # df_sitting_upsampled = resample(df[df['class']=='sitting_lying_down'], 
        #                                 replace=True,     # sample with replacement
        #                                 n_samples=num_majority ,    # to match majority class
        #                                 random_state=42) # reproducible results
        # # Combine majority class with upsampled minority class
        # df_upsampled = pd.concat([df_moving, df_standing_upsampled, df_sitting_upsampled])

        # Display new class counts
        #df_upsampled['class'].value_counts()
        # df_train = df_upsampled.copy()

        df.to_csv(os.path.join(self.out_dir, filename), index=False)

        print("\nPreprocssed {} data saved to {}".format(data_type, os.path.join(self.out_dir, filename)))
            
            
def factory(dataset_path, version):
    """Define dataset type and split training and validation"""
    # Load NuScenes data class
    nusc = NuScenes(version=version, dataroot=os.path.join(dataset_path,version), verbose=True)
    scenes = nusc.scene

    # Split scenes
    split_scenes = create_splits_scenes()
    if version == 'v1.0-mini':
        split_train, split_val, split_test = split_scenes['train'], split_scenes['val'], split_scenes['test']
    else:
        # Train/val/test split based on 850 fully annotated images from TrainVal NuScenes
        split_train, split_val, split_test = split_scenes['train_annotated'], split_scenes['val_annotated'], split_scenes['test_annotated']

    return nusc, scenes, split_train, split_val, split_test


        
if __name__ == '__main__':
    args = parse_args()
    raw_data = ['train', 'test', 'val']
    nu_preprocessor = NuscenesPreprocessor( dataset_path=args.data_dir, 
                                            version=args.version, 
                                            input=args.input,
                                            output=args.output)

    for data_type in raw_data:
        print('Reading {} data...'.format(data_type))
        nu_preprocessor.preprocess(data_type)