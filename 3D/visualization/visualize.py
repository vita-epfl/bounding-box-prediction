"""
Created on Sat Nov 27 23:06:36 2021

@author: Celinna
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import imageio
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.pyplot import AutoLocator
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box



K  = [[1158, 0, 960], [0, 1158, 540], [0, 0, 1]] # JTA internal camera parameters mat

coords = {
    'p1': ['x1','y1','z1'], # left-bottom-front
    'p2': ['x1','y2','z1'], #left-top-front
    'p3': ['x2','y2','z1'], #right-top-front
    'p4': ['x2','y1','z1'], #right-bottom-front
    'p5': ['x1','y1','z2'], #left-bottom-back
    'p6': ['x1','y2','z2'], #left-top-back
    'p7': ['x2','y2','z2'], #right-top-back
    'p8': ['x2','y1','z2'] #right-bottom-back
}


class visualize():
    '''
    Create GIF

    Params
    ------
    idx (int):      from all the frames with pedestrian with pid, the index of the frame to start from
                    - usually set to 0 to see first appearance of pedestrian
    pid (int):      pedestrian id (starting from 0)
    true (df):      contains true bbox values
    pred (df):      contains predicted bbox values
    gif_name (str): name of gif to be saved
    args (class):   arguments for training
    outpath (str):  location to save gif
    '''
    
    def __init__(self, dataset):
        self.data = dataset
        if self.data == 'nuscenes':
            self.nusc = NuScenes(version='v1.0-trainval', dataroot='/work/vita/datasets/NuScenes_full/US', verbose=True)

    
    def get_gif(self, idx, df_true, df_pred, args, outpath, save=False, custom_name=None):
        """
        Get the GIF
        """
        if self.data == 'JTA':
            cols = ['x','y','z','w','h','d']

            inpath = df_true.scenefolderpath.iloc[idx][0] # get folder path

            # Get frame numbers  (Current line stores frame numbers of obs, not prediction)
            filenames = df_true.filename.iloc[idx]
            frames = [int(float(os.path.splitext(x)[0]))+args.input for x in filenames] # must add input size to all frames
            pred_filenames = [str(x)+'.jpg' for x in frames] # prediction file names
            data =  df_true.loc[(df_true.filename.str[0] == filenames[0]) & (df_true.scenefolderpath.str[0] == inpath)]
            print(data.shape)
            all_pids = np.asarray(data.ID)

            # Get 5 closest pedestrians to camera in a specific frame
            path = args.dataset
            file = os.path.basename(os.path.normpath(inpath))
            temp = pd.read_csv(join(path, join(args.dtype,file) +'.csv'))
            closest_pids = get_closest_peds(temp, frames[0], list(all_pids))
            plot_pids = []

            # plot boxes for closest pids
            all_obs = [] # list of dataframes of all pedestrians
            all_preds = []

            for pid in closest_pids:
                temp = data[data.ID == pid]
                all_obs.append(cam2pix_bbox(pd.DataFrame(temp.future_bounding_box.item(), columns=cols))) # get observations
                all_preds.append(cam2pix_bbox(pd.DataFrame(df_pred[temp.index[0]], columns=cols))) # get predictions
                plot_pids.append(pid)

            print('Plotting pedestrians with pids: {}'.format(plot_pids))

            if len(plot_pids) == 0:
                return 'Not enough pedestrians in frames or all pedestrians are obscured'

            # plot 3d bbox on 2d images
            for i in range(args.output): # number of frames to annotate
                im = Image.open(join(inpath, pred_filenames[i])) #always plotting first 16 frames 
                fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
                for k in range(len(plot_pids)): 
                    obs = all_obs[k].iloc[i] # get data for specific frame
                    preds = all_preds[k].iloc[i]
                    draw_3Dbbox(ax, obs, preds)
                ax.axis('off')
                plt.imshow(im)
                if save:
                    name = join(outpath, str(frames[i]) + '_temp.jpg') 
                    plt.savefig(name, bbox_inches='tight')
                    plt.close()
        
        elif self.data == 'nuscenes':
            # get scene
            ann_token = df_true['ann_token'][idx][0]
            ann = self.nusc.get('sample_annotation', ann_token)
            sample = self.nusc.get('sample', ann['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            file = scene['name']

            # plot annotations
            frames = []
            for i in range(args.output):
                ann_token = df_true['ann_token'][idx][i+4] # start from first prediction frame
                cam_token = df_true['cam_token'][idx][i+4]

                # Plot CAMERA view.
                data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
                true_box = boxes[0]

                # original plotting data takes wlh, prediction in whd
                xyz = df_pred[idx][i][0:3]
                whd = df_pred[idx][i][3:6]
                wlh = [whd[i] for i in (0, 2, 1)] 

                pred_box = Box(center= xyz, size=wlh, orientation=true_box.orientation)

                fig, axes = plt.subplots(figsize=(9, 9), dpi=100)
                im = Image.open(data_path)
                axes.imshow(im)
                axes.set_title(self.nusc.get('sample_data', cam_token)['channel'])
                axes.axis('off')
                c_true = ('g', 'g', 'g')
                c_pred = ('r', 'r', 'r')
                true_box.render(axes, view=camera_intrinsic, normalize=True, colors=c_true)
                pred_box.render(axes, view=camera_intrinsic, normalize=True, colors=c_pred)
                frames.append(i)

                if save:
                    name = join(outpath, str(frames[i]) + '_temp.jpg') 
                    plt.savefig(name, bbox_inches='tight')

                plt.show()
                plt.close()

        else:
            print('Error: unknown dataset!')

        if save:
            # save files as a gif
            if custom_name:
                gif_name = '{}_{}_frame{}_idx{}_{}.gif'.format(args.dtype, file, str(frames[0]), idx, custom_name)
            else:
                gif_name = '{}_{}_frame{}_idx{}.gif'.format(args.dtype, file, str(frames[0]), idx)

            with imageio.get_writer(join(outpath, gif_name), mode='I') as writer:
                for k in frames:
                    name = str(k) + '_temp.jpg'
                    image = imageio.imread(join(outpath, name))
                    writer.append_data(image)
#                     os.remove(join(outpath, name))
            print('Saved GIF as {}'.format(gif_name))
        plt.clf()
    
        
def cam2pix_bbox(df):
    """
    Turns 3D coordinates from camera to pixel view
    """
    bbox = pd.DataFrame()
    temp = pd.DataFrame()

    temp['x1'] = df.x - df.w/2
    temp['y1'] = df.y - df.h/2
    temp['z1'] = df.z - df.d/2
    temp['x2'] = df.x + df.w/2
    temp['y2'] = df.y + df.h/2
    temp['z2'] = df.z + df.d/2

    for point in coords.keys():
        coord = coords[point]
        x_p = temp[coord[0]] / temp[coord[2]]
        y_p = temp[coord[1]] / temp[coord[2]]
        px = (K[0][0] * x_p + K[0][2]).astype(int)
        py = (K[1][1] * y_p + K[1][2]).astype(int)
        #px = np.where(px < 0, 0, px)
        #py = np.where(py < 0, 0, py)

        bbox[point] = list(zip(px, py))
        
    return bbox


def get_line_coords(p1, p2):
    '''
    Get coordinates in required format for line plotting in matplotlib

    [x1, x2], [y1, y2]
    '''
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    P = np.vstack((p1,p2))
    P = np.transpose(P)

    return P


def get_lines(temp):
    '''
    Get lines for 3D bbox in 2D plots
    '''
    l1 =    get_line_coords(temp.p1, temp.p2) 
    l2 =    get_line_coords(temp.p2, temp.p3)
    l3 =    get_line_coords(temp.p3, temp.p4)
    l4 =    get_line_coords(temp.p1, temp.p4)
    l5 =    get_line_coords(temp.p5, temp.p6)
    l6 =    get_line_coords(temp.p6, temp.p7)
    l7 =    get_line_coords(temp.p7, temp.p8)
    l8 =    get_line_coords(temp.p8, temp.p5)
    l9 =    get_line_coords(temp.p1, temp.p5)
    l10 =   get_line_coords(temp.p2, temp.p6)
    l11 =   get_line_coords(temp.p3, temp.p7)
    l12 =   get_line_coords(temp.p4, temp.p8)

    L = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]

    return L


def draw_3Dbbox(ax, df_true, df_pred):
    '''
    Draw obs and pred 3D bbox on 2D image
    '''
    L_true = get_lines(df_true)
    L_pred = get_lines(df_pred)

    for i in range(len(L_true)):
        ax.plot(L_true[i][0], L_true[i][1], color='g', linewidth=1.5)
        ax.plot(L_pred[i][0], L_pred[i][1], color='r', linewidth=1.5)

    
def get_closest_peds(df, frame, pids, n=5):
    """
    Get the closest pedestrians in the given index frame camera location
    """
    # check specific frame
    df_frame = df.loc[(df.frame==frame) & (df.ID.isin(pids))]
    temp = cam2pix_bbox(df_frame)
    bbox_size = []
    
    # got through rows and get relative size of bounding box
    for index, row in temp.iterrows():
        w = LA.norm(np.array(row.p2) - np.array(row.p1)).astype(int)
        h = LA.norm(np.array(row.p2) - np.array(row.p3)).astype(int)
        d = LA.norm(np.array(row.p2) - np.array(row.p7)).astype(int) 

        total = np.sum([w, h, d])
        bbox_size.append(total)

    bbox_size = np.array(bbox_size)
    
    # get index locations of num_peds largest bounding boxes
    if df_frame.shape[0] < n:
        n = df_frame.shape[0]
    
    indexes = np.argsort(bbox_size) #get index of closest peds
    indexes = indexes[::-1][:n]
    pids = np.asarray(df_frame.ID)[indexes].astype(int) # get corresponding pid
    
    return pids
                
    
def get_3d_vertices(df):
    """
    Get 3D vertices (not used)
    """
    temp = pd.DataFrame()

    #switch order of y and z
    temp['x1'] = df.x - df.w/2
    temp['z1'] = df.y - df.h/2 
    temp['y1'] = df.z - df.d/2
    temp['x2'] = df.x + df.w/2
    temp['z2'] = df.y + df.h/2
    temp['y2'] = df.z + df.d/2

    return temp


def get_axlim(ax_min, ax_max, row):
    """
    Get the limits for the 3D plot
    """
    ax_max[0] = max(row['x2'], ax_max[0])
    ax_max[1] = max(row['y2'], ax_max[1])
    ax_max[2] = max(row['z2'], ax_max[2])
    ax_min[0] = min(row['x1'], ax_min[0])
    ax_min[1] = min(row['y1'], ax_min[1])
    ax_min[2] = min(row['z1'], ax_min[2])

    return ax_min, ax_max


def set_plot_axes(ax, ax_min, ax_max):
    """
    Set 3D plot axes
    """
    ax.set_xlim(xmin=ax_min[0], xmax=ax_max[0])
    ax.set_ylim(ymin=ax_min[1], ymax=ax_max[1])
    ax.set_zlim(zmin=ax_min[2], zmax=ax_max[2])

    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.zaxis.set_major_locator(AutoLocator())

    ax.set_aspect('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    
def get_colormap(n):
    """
    Gets colormaps
    """
    colors=np.random.rand(n)
    #cmap=plt.cm.RdYlBu_r
    cmap=plt.cm.jet
    c=cmap(colors)
    
    return c
    
    
def plot_3d(df, num_peds, color, view='default'):
    """
    Make 3D plots
    """
    bbox = get_3d_vertices(df)

    # draw figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection='3d')

    if view == 'top':
        ax.view_init(elev=90, azim=180) #birds eye view

    ax_min = np.ones(3)*np.inf
    ax_max = -np.ones(3)*np.inf
    for i in range(num_peds):
        Z = []
        for point in coords.keys():
            p = coords[point]
            Z.append([bbox[p[0]][i], bbox[p[1]][i], bbox[p[2]][i]])

        Z = np.asarray(Z)
        verts = [[Z[0],Z[1],Z[2],Z[3]], [Z[4],Z[5],Z[6],Z[7]], [Z[0],Z[1],Z[5],Z[4]],
                 [Z[2],Z[3],Z[7],Z[6]], [Z[1],Z[2],Z[6],Z[5]], [Z[4],Z[7],Z[3],Z[0]]]

        ax.add_collection3d(Poly3DCollection(verts, facecolors=color[i], linewidths=1, edgecolors=color[i], alpha=.20))

        ax_min, ax_max = get_axlim(ax_min, ax_max, bbox.iloc[i])

    # set plot ax params
    set_plot_axes(ax, ax_min, ax_max)
    plt.show()