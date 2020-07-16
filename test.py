import os
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud
from models.D3Feat import KPFCNN
from utils.timer import Timer, AverageMeter
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence


def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'D3Feat')
                target_desc = get_desc(descpath, cloud_bin_t, 'D3Feat')
                source_score = get_scores(scorepath, cloud_bin_s, 'D3Feat').squeeze()
                target_score = get_scores(scorepath, cloud_bin_t, 'D3Feat').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)
                
                # randomly select 5000 keypts
                if args.random_points:
                    source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(source_score)[-args.num_points:]
                    target_indices = np.argsort(target_score)[-args.num_points:]
                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]
                
                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    
    # generate descriptors
    recall_list = []
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"{config.root}/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()
            features, scores = model(inputs)
            pcd_size = inputs['stack_lengths'][0][0]
            pts = inputs['points'][0][:int(pcd_size)]
            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
            # scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat', features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=250, type=int)
    parser.add_argument('--generate_features', default=False, action='store_true')
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO, 
        filename=log_filename, 
        filemode='w', 
        format="")


    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model 
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-1):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('last_unary')

    # # dynamically load the model from snapshot
    # module_file_path = f'snapshot/{chosen_snap}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # model = module.KPFCNN(config)

    model = KPFCNN(config)
    model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    model.eval()


    save_path = f'geometric_registration/{args.chosen_snapshot}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.generate_features:
        dset = ThreeDMatchTestset(root=config.root,
                            config=config,
                            last_scene=False,
                        )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    )
        generate_features(model.cuda(), dloader, config, args.chosen_snapshot)

    # register each pair of fragments in scenes using multiprocessing.
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
    return_dict = Manager().dict()
    # register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    jobs = []
    for scene in scene_list:
        p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()

    recalls = [v[0] for k, v in return_dict.items()]
    inlier_nums = [v[1] for k, v in return_dict.items()]
    inlier_ratios = [v[2] for k, v in return_dict.items()]

    logging.info("*" * 40)
    logging.info(recalls)
    logging.info(f"All 8 scene, average recall: {np.mean(recalls):.2f}%")
    logging.info(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    logging.info(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios)*100:.2f}%")