import os
import os.path
from os.path import join, exists
import numpy as np
import json
import pickle
import random
import open3d as o3d
from utils.pointcloud import make_point_cloud
import torch.utils.data as data
from scipy.spatial.distance import cdist


def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz
    
def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T

    
class ThreeDMatchDataset(data.Dataset):
    __type__ = 'descriptor'
    
    def __init__(self, 
                 root, 
                 split='train', 
                 num_node=16, 
                 downsample=0.03, 
                 self_augment=False, 
                 augment_axis=1, 
                 augment_rotation=1.0,
                 augment_translation=0.001,
                 config=None,
                 ):
        self.root = root
        self.split = split
        self.num_node = num_node
        self.downsample = downsample
        self.self_augment = self_augment
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.config = config

        assert self_augment == False
        
        # containers
        self.ids = []
        self.points = []
        self.src_to_tgt = {}
        
        # load data
        pts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_points.pkl')
        keypts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_keypts.pkl')

        if exists(pts_filename) and exists(keypts_filename):
            with open(pts_filename, 'rb') as file:
                data = pickle.load(file)
                self.points = [*data.values()]
                self.ids_list = [*data.keys()]
            with open(keypts_filename, 'rb') as file:
                self.correspondences = pickle.load(file)
        else:
            print("PKL file not found.")
            return

        for idpair in self.correspondences.keys():
            src = idpair.split("@")[0]
            tgt = idpair.split("@")[1]
            # add (key -> value)  src -> tgt 
            if src not in self.src_to_tgt.keys():
                self.src_to_tgt[src] = [tgt]
            else:
                self.src_to_tgt[src] += [tgt]

    def __getitem__(self, index):
        src_id = list(self.src_to_tgt.keys())[index]
        
        if random.random() > 0.5:
            tgt_id = self.src_to_tgt[src_id][0]
        else:
            tgt_id = random.choice(self.src_to_tgt[src_id])
            
        src_ind = self.ids_list.index(src_id)
        tgt_ind = self.ids_list.index(tgt_id)
        src_pcd = make_point_cloud(self.points[src_ind])
        if self.self_augment:
            tgt_pcd = make_point_cloud(self.points[src_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[tgt_ind].shape[0]
        else:
            tgt_pcd = make_point_cloud(self.points[tgt_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[tgt_ind].shape[0]
        if N_src > 50000 or N_tgt > 50000:
            return self.__getitem__(int(np.random.choice(self.__len__(), 1)))

        # data augmentation
        gt_trans = np.eye(4).astype(np.float32)
        R = rotation_matrix(self.augment_axis, self.augment_rotation)
        T = translation_matrix(self.augment_translation)
        gt_trans[0:3, 0:3] = R
        gt_trans[0:3, 3] = T
        tgt_pcd.transform(gt_trans)
        

        corr = self.correspondences[f"{src_id}@{tgt_id}"]
        sel_corr = corr[np.random.choice(len(corr), self.num_node, replace=False)]
        
        sel_P_src = np.array(src_pcd.points)[sel_corr[:,0], :].astype(np.float32)
        sel_P_tgt = np.array(tgt_pcd.points)[sel_corr[:,1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)
        # sel_P_src = np.array(src_pcd.points)[sel_src, :].astype(np.float32)
        # sel_P_tgt = np.array(tgt_pcd.points)[sel_tgt, :].astype(np.float32)
                
        pts0 = np.array(src_pcd.points).astype(np.float32)
        pts1 = np.array(tgt_pcd.points).astype(np.float32)
        feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)
        
        return pts0, pts1, feat0, feat1, sel_corr, dist_keypts
            
    def __len__(self):
        return len(self.src_to_tgt.keys())

class ThreeDMatchTestset(data.Dataset):
    __type__ = 'descriptor'
    def __init__(self, 
                root, 
                downsample=0.03, 
                config=None,
                last_scene=False,
                ):
        self.root = root
        self.downsample = downsample
        self.config = config
        
        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if last_scene == True:
            self.scene_list = self.scene_list[-1:]
        for scene in self.scene_list:
            self.test_path = f'{self.root}/fragments/{scene}'
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = o3d.io.read_point_cloud(join(self.test_path, ind))
                pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.03)
                
                # Load points and labels
                points = np.array(pcd.points)

                self.points += [points]
                self.ids_list += [scene + '/' + ind]
        return

    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return self.num_test

if __name__ == "__main__":
    dset = ThreeDMatchTestset(root='/home/xybai/KPConv/data/3DMatch/')
    import pdb
    pdb.set_trace()