import numpy as np
import torch
import random
import json
import os


def ADE(pred, true, is_3D=False):
    if not is_3D:
        displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2)
    else:
        displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2\
                                  + (pred[:,:,2]-true[:,:,2])**2)
        
    ade = torch.mean(displacement)
    
    return ade


def FDE(pred, true, is_3D=False):
    if not is_3D:
        displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2)
    else:
        displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2\
                                  + (pred[:,-1,2]-true[:,-1,2])**2)
    
    fde = torch.mean(displacement)
    
    return fde


def AIOU(pred, true, is_3D=False):
    if not is_3D:
        min_pred = pred[:,:,:2]-pred[:,:,2:]/2
        max_pred = pred[:,:,:2]+pred[:,:,2:]/2
        min_true = true[:,:,:2]-true[:,:,2:]/2
        max_true = true[:,:,:2]+true[:,:,2:]/2

        min_inter = torch.max(min_pred, min_true)
        max_inter = torch.min(max_pred, max_true)

        interArea = torch.max(torch.zeros(min_inter.shape[0],min_inter.shape[1]).to('cuda'), (max_inter[:,:,0]-min_inter[:,:,0])) *\
                    torch.max(torch.zeros(max_inter.shape[0],max_inter.shape[1]).to('cuda'), (max_inter[:,:,1]-min_inter[:,:,1]))

        pred_a = pred[:,:,2] * pred[:,:,3]
        true_a = true[:,:,2] * true[:,:,3]

        iou = torch.mean(interArea / (pred_a + true_a - interArea))
    else:
        min_pred = pred[:,:,:3]-pred[:,:,3:]/2
        max_pred = pred[:,:,:3]+pred[:,:,3:]/2
        min_true = true[:,:,:3]-true[:,:,3:]/2
        max_true = true[:,:,:3]+true[:,:,3:]/2

        min_inter = torch.max(min_pred, min_true)
        max_inter = torch.min(max_pred, max_true)

        interArea = torch.max(torch.zeros(min_inter.shape[0],min_inter.shape[1]).to('cuda'), (max_inter[:,:,0]-min_inter[:,:,0])) *\
                    torch.max(torch.zeros(max_inter.shape[0],max_inter.shape[1]).to('cuda'), (max_inter[:,:,1]-min_inter[:,:,1])) *\
                    torch.max(torch.zeros(max_inter.shape[0],max_inter.shape[1]).to('cuda'), (max_inter[:,:,2]-min_inter[:,:,2]))

        pred_a = pred[:,:,3] * pred[:,:,4] * pred[:,:,5]
        true_a = true[:,:,3] * true[:,:,4] * true[:,:,5]

        iou = torch.mean(interArea / (pred_a + true_a - interArea))
        
    return float(iou)


def FIOU(pred, true, is_3D=False):
    if not is_3D:
        min_pred = pred[:,-1,:2]-pred[:,-1,2:]/2
        max_pred = pred[:,-1,:2]+pred[:,-1,2:]/2
        min_true = true[:,-1,:2]-true[:,-1,2:]/2
        max_true = true[:,-1,:2]+true[:,-1,2:]/2

        min_inter = torch.max(min_pred, min_true)
        max_inter = torch.min(max_pred, max_true)

        interArea = torch.max(torch.zeros(min_inter.shape[0]).to('cuda'), (max_inter[:,0]-min_inter[:,0])) * \
                    torch.max(torch.zeros(max_inter.shape[0]).to('cuda'), (max_inter[:,1]-min_inter[:,1]))

        pred_a = pred[:,-1,2] * pred[:,-1,3]
        true_a = true[:,-1,2] * true[:,-1,3]

        iou = torch.mean(interArea / (pred_a + true_a - interArea))
       
    else:
        min_pred = pred[:,-1,:3]-pred[:,-1,3:]/2
        max_pred = pred[:,-1,:3]+pred[:,-1,3:]/2
        min_true = true[:,-1,:3]-true[:,-1,3:]/2
        max_true = true[:,-1,:3]+true[:,-1,3:]/2

        min_inter = torch.max(min_pred, min_true)
        max_inter = torch.min(max_pred, max_true)

        interArea = torch.max(torch.zeros(min_inter.shape[0]).to('cuda'), (max_inter[:,0]-min_inter[:,0])) * \
                    torch.max(torch.zeros(max_inter.shape[0]).to('cuda'), (max_inter[:,1]-min_inter[:,1])) * \
                    torch.max(torch.zeros(max_inter.shape[0]).to('cuda'), (max_inter[:,2]-min_inter[:,2]))

        pred_a = pred[:,-1,3] * pred[:,-1,4] * pred[:,-1,5]
        true_a = true[:,-1,3] * true[:,-1,4] * true[:,-1,5]

        iou = torch.mean(interArea / (pred_a + true_a - interArea))
        
    return float(iou)


def compute_center(row, is_3D=False):
    if not is_3D:
        row['x'] = row['x'] + row['w']/2
        row['y'] = row['y'] + row['h']/2
        
    else:
        row['x'] = row['x'] + row['w']/2
        row['y'] = row['y'] + row['h']/2
        row['z'] = row['z'] + row['d']/2
    
    return row


def speed2pos(preds, obs_p, is_3D=False):
    if not is_3D:
        pred_pos = torch.zeros(preds.shape[0], preds.shape[1], 4).to('cuda')
        current = obs_p[:,-1,:]
        for i in range(preds.shape[1]):
            pred_pos[:,i,:] = current + preds[:,i,:]
            current = pred_pos[:,i,:]
            
        pred_pos[:,:,0] = torch.min(pred_pos[:,:,0], 1920*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,1] = torch.min(pred_pos[:,:,1], 1080*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,0] = torch.max(pred_pos[:,:,0], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,1] = torch.max(pred_pos[:,:,1], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
    else:
        pred_pos = torch.zeros(preds.shape[0], preds.shape[1], 6).to('cuda')
        current = obs_p[:,-1,:]
        for i in range(preds.shape[1]):
            pred_pos[:,i,:] = current + preds[:,i,:]
            current = pred_pos[:,i,:]
            
        pred_pos[:,:,0] = torch.min(pred_pos[:,:,0], 100*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,1] = torch.min(pred_pos[:,:,1], 100*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,0] = torch.max(pred_pos[:,:,0], -100*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,1] = torch.max(pred_pos[:,:,1], -100*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
            
    return pred_pos


    
def check_continuity(my_list, skip):
    '''
    Checks if there are frames continuously
    Returns True is there is discontinuity
    '''
    return any(a+skip != b for a, b in zip(my_list, my_list[1:]))


def get_unique_tokens(list_fin):
    """
    list of json files --> list of unique scene tokens
    """
    list_token_scene = []

    # Open one json file at a time
    for name_fin in list_fin:
        with open(name_fin, 'r') as f:
            dict_fin = json.load(f)

        # Check if the token scene is already in the list and if not add it
        if dict_fin['token_scene'] not in list_token_scene:
            list_token_scene.append(dict_fin['token_scene'])

    return list_token_scene


def split_scenes(list_token_scene, train, val, dir_main, save=False, load=True):
    """
    Split the list according tr, val percentages (test percentage is a consequence) after shuffling the order
    """

    path_split = os.path.join(dir_main, 'scenes', 'split_scenes.json')

    if save:
        random.seed(1)
        random.shuffle(list_token_scene)  # it shuffles in place
        n_scenes = len(list_token_scene)
        n_train = round(n_scenes * train / 100)
        n_val = round(n_scenes * val / 100)
        list_train = list_token_scene[0: n_train]
        list_val = list_token_scene[n_train: n_train + n_val]
        list_test = list_token_scene[n_train + n_val:]

        dic_split = {'train': list_train, 'val': list_val, 'test': list_test}
        with open(path_split, 'w') as f:
            json.dump(dic_split, f)

    if load:
        with open(path_split, 'r') as f:
            dic_split = json.load(f)

    return dic_split


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