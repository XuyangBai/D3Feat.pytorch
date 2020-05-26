import os
import open3d as o3d
import numpy as np

def build_correspondence(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """

    distance = np.sqrt(2 - 2 * (source_desc @ target_desc.T))
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    target_idx = np.argmin(distance, axis=0)
    target_dis = np.min(distance, axis=0)

    result = []
    for i in range(len(source_idx)):
        if target_idx[source_idx[i]] == i:
            result.append([i, source_idx[i]])
    return np.array(result)


def get_pcd(pcdpath, filename):
    return o3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_keypts(keyptspath, filename):
    keypts = np.load(os.path.join(keyptspath, filename + f'.npy'))
    return keypts


def get_desc(descpath, filename, desc_name):
    desc = np.load(os.path.join(descpath, filename + f'.{desc_name}.npy'))
    return desc


def get_scores(descpath, filename, desc_name):
    scores = np.load(os.path.join(descpath, filename + '.npy'))
    return scores


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result
