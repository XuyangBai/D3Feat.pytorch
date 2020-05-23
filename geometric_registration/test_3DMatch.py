import sys
import open3d
import numpy as np
import time
import os
import importlib
from utils.pointcloud import make_point_cloud
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, loadlog
import cv2
import torch
from functools import partial

def build_correspondence(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """

    source_idx = []
    source_dis = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    match = bf_matcher.match(source_desc, target_desc)
    for match_val in match:
        source_idx.append(match_val.trainIdx)
        source_dis.append(match_val.distance)
    target_idx = []
    target_dis = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    match = bf_matcher.match(target_desc, source_desc)
    for match_val in match:
        target_idx.append(match_val.trainIdx)
        target_dis.append(match_val.distance)

    result = []
    for i in range(len(source_idx)):
        if target_idx[source_idx[i]] == i:
            result.append([i, source_idx[i]])
    return np.array(result)

def register2Fragments(id1, id2, keyptspath, descpath, resultpath, logpath, gtLog, desc_name, inlier_ratio, distance_threshold):
    """
    Register point cloud {id1} and {id2} using the keypts location and descriptors.
    """
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        return 0, 0, 0
    source_keypts = get_keypts(keyptspath, cloud_bin_s)
    target_keypts = get_keypts(keyptspath, cloud_bin_t)
    source_desc = get_desc(descpath, cloud_bin_s, desc_name)
    target_desc = get_desc(descpath, cloud_bin_t, desc_name)
    source_desc = np.nan_to_num(source_desc)
    target_desc = np.nan_to_num(target_desc)
    # Select {num_keypts} points based on the scores. The descriptors and keypts are already sorted based on the detection score.
    # num_keypts = 250
    # source_keypts = source_keypts[-num_keypts:, :]
    # source_desc = source_desc[-num_keypts:, :]
    # target_keypts = target_keypts[-num_keypts:, :]
    # target_desc = target_desc[-num_keypts:, :]
    # Select {num_keypts} points randomly.
    num_keypts = 5000
    source_indices = np.random.choice(range(source_keypts.shape[0]), num_keypts)
    target_indices = np.random.choice(range(target_keypts.shape[0]), num_keypts)
    source_keypts = source_keypts[source_indices, :]
    source_desc = source_desc[source_indices, :]
    target_keypts = target_keypts[target_indices, :]
    target_desc = target_desc[target_indices, :]
    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        # skip the pairs that have less than 30% overlap.
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        # build correspondence set in feature space.
        corr = build_correspondence(source_desc, target_desc)

        # calculate the inlier ratio, this is for Feature Matching Recall.
        gt_trans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gt_trans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < distance_threshold)
        if num_inliers / len(distance) < inlier_ratio:
            print(key)
            print("num_corr:", len(corr), "inlier_ratio:", num_inliers / len(distance))
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

    # write the result into resultpath so that it can be re-shown.
    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag
  
def deal_with_one_scene(inlier_ratio, distance_threshold, scene):
    """
    Function to register all the fragments pairs in one scene.
    """
    logpath = f"log_result/{scene}-evaluation"
    pcdpath = f"/home/xybai/KPConv/data/3DMatch/fragments/{scene}/"
    keyptspath = f"{desc_name}_{timestr}/keypoints/{scene}"
    descpath = f"{desc_name}_{timestr}/descriptors/{scene}"
    gtpath = f'gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    resultpath = f"pred_result/{scene}/{desc_name}_result_{timestr}"
    if not os.path.exists(f"pred_result/{scene}/"):
        os.mkdir(f"pred_result/{scene}/")
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    if not os.path.exists(logpath):
        os.mkdir(logpath)

    # register each pair
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    print(f"Start Evaluate Descriptor {desc_name} for {scene}")
    start_time = time.time()
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            register2Fragments(id1, id2, keyptspath, descpath, resultpath, logpath, gtLog, desc_name, inlier_ratio, distance_threshold)
    print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")

def read_register_result(resultpath, id1, id2):
    """
    Read the registration result of {id1} & {id2} from the resultpath
    Return values contain the inlier_number, inlier_ratio, flag(indicating whether this pair is a ground truth match).
    """
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums

def calculate_features(model, dataloader, scene_list):
    """ calculate the dense feature descriptors for each point cloud in the test set. 
    """
    model.eval()
    descriptor_path = f'D3Feat_{timestr}/descriptors'
    keypoint_path = f'D3Feat_{timestr}/keypoints'
    score_path = f'D3Feat_{timestr}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)
        
    dataloader_iter = dataloader.__iter__()
    for scene in scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"/home/xybai/KPConv/data/3DMatch/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()
            output = model(inputs)
            pcd_size = inputs['stack_lengths'][0][0]
            pts = inputs['points'][0][:int(pcd_size)]
            features = output[:int(pcd_size)]
            scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat', features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")
    
if __name__ == '__main__':
    ## Before run this script, you should copy the kernels/k_015_center.ply into this directory.
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    desc_name = 'D3Feat'
    timestr = sys.argv[1]
    
    # dynamically load the model from snapshot
    module_file_path = f'/home/xybai/KPConv_PyTorch/models/KPFCNN_desc.py'
    module_name = 'models'
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    
    module_file_path = f'/home/xybai/KPConv_PyTorch/snapshot/3DMatch_KPConvNet{timestr}/train.py'
    module_name = 'train'
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config_module)
    
    model = module.KPFCNN(config_module.ThreeDMatchConfig()).cuda()
    model.load_state_dict(torch.load(f'/home/xybai/KPConv_PyTorch/snapshot/3DMatch_KPConvNet{timestr}/models/model_90.pth')['state_dict'])
    model.eval()
        
    # calculate features for each point cloud using the pre-trained model 
    save_path = f'D3Feat_{timestr}'
    last_scene = False
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        dset = ThreeDMatchTestset(root='/home/xybai/KPConv/data/3DMatch', config=config_module.ThreeDMatchConfig(), last_scene=last_scene)
        dataloader = get_dataloader(dset, batch_size=1, shuffle=False)
        scene_list = dset.scene_list
        calculate_features(model, dataloader, scene_list) 
    else:
        print("Descriptors already exists.")
    
    # register each pair of fragments in scenes.
    from multiprocessing import Pool

    pool = Pool(len(scene_list))
    func = partial(deal_with_one_scene, 0.05, 0.10)
    pool.map(func, scene_list)
    pool.close()
    pool.join()
    
    # collect all the data and print the results.
    inliers_list = []
    recall_list = []
    inliers_ratio_list = []
    pred_match = 0
    gt_match = 0
    for scene in scene_list:
        # evaluate
        pcdpath = f"/home/xybai/KPConv/data/3DMatch/fragments/{scene}/"
        resultpath = os.path.join(".", f"pred_result/{scene}/{desc_name}_result_{timestr}")
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(resultpath, id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])  # inlier_number, inlier_ratio, flag.
        result = np.array(result)
        gt_results = np.sum(result[:, 2] == 1)
        pred_results = np.sum(result[:, 1] > 0.05)
        pred_match += pred_results
        gt_match += gt_results
        recall = float(pred_results / gt_results) * 100
        print(f"Correct Match {pred_results}, ground truth Match {gt_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 2] == 1, result[:, 0], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliners: {ave_num_inliers}")
        ave_inlier_ratio = np.sum(np.where(result[:, 2] == 1, result[:, 1], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliner Ratio: {ave_inlier_ratio}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
        inliers_ratio_list.append(ave_inlier_ratio)

    print("*" * 40)
    print(recall_list)
    # print(f"True Avarage Recall: {pred_match / gt_match * 100}%")
    print(f"Matching Recall Std: {np.std(recall_list)}")
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
    average_inliers_ratio = sum(inliers_ratio_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers ratio: {average_inliers_ratio}")