import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

from src.utils import config
from src.utils.data_loader import BreakfastDataset, SaladsDataset
from src.core import abd
from src.metrics.accuracy import calculate_mof, GlobalMoF, calculate_f1
from src.utils.visualization import plot_online_abd_results


def run_online_mode(dataset_name: str, boundaries_type: str = "eval", log: bool = False, visual: bool = False):
    
    dataset_conf = config.getConfigYAML("config/dataset.yaml")
    
    if dataset_name == "breakfast":
        br_conf = dataset_conf["datasets"]['breakfast']
        dataset = BreakfastDataset(**br_conf).getDataset()
    elif dataset_name == "50salads":
        br_conf = dataset_conf["datasets"]['50salads']
        dataset = SaladsDataset(br_conf['name'], br_conf['dataset_path'], br_conf["boundaries"][boundaries_type]['threshold_classes'])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in online_mode")

    L = br_conf["boundaries"][boundaries_type]['window_size']
    
    if log:
        logger = config.getLogger("production")
        
    print(f"Running ONLINE mode on {dataset_name} dataset with window L={L}...")
    tq = tqdm(dataset, desc=f"Processing streaming on {dataset_name}", unit="video")

    history = {
        "video_id": [],
        "MoF": [],
        "F1": []
    }

    global_mof = GlobalMoF()
    semantic_gate = 0.9987 if dataset_name == "breakfast" else 0.99998

    to_visual = 0.75 if dataset_name == "breakfast" else 0.6

    for item in tq:
        time_s = time.time()

        features = item['video_feature']
        video_label = item['video_label']
        video_id = item['video_id']
        
        video_feature = torch.tensor(features, dtype=torch.float32)
        target_len = video_feature.shape[0]

        processor = abd.OnlineABDProcessor(kernel_size=L, window_size=L, semantic_gate_value=semantic_gate)
        
        for i in range(target_len):
            processor.step(video_feature[i])
            
        b_list = sorted(list(set(processor.boundaries)))
        if len(b_list) == 0 or b_list[-1] != target_len - 1:
            b_list.append(target_len - 1)
            
        preds = np.zeros(target_len, dtype=int)
        start_idx = 0
        seg_id = 0
        
        for end_idx in b_list:
            preds[start_idx : end_idx + 1] = seg_id
            seg_id += 1
            start_idx = end_idx + 1

        gt_arr = np.array(video_label)

        if len(preds) != target_len:
            preds = zoom(preds, target_len / len(preds), order=0)
        if len(gt_arr) != target_len:
            gt_arr = zoom(gt_arr, target_len / len(gt_arr), order=0)

        mof_video, remapped_preds = calculate_mof(preds, gt_arr)
        global_mof.update(remapped_preds, gt_arr)
        f1 = calculate_f1(remapped_preds, gt_arr)

        if log:
            logger.info(f"Online Video {video_id} | MoF: {mof_video*100:.2f}% | F1: {f1*100:.2f}% | Time: {(time.time()-time_s):.2f}s")
            
        tq.set_postfix_str(f"MoF: {mof_video*100:.2f}% - F1: {f1*100:.2f}%")
        
        if f1 > to_visual and visual:
            plot_online_abd_results(
                similarities=processor.similarities,
                thresholds=processor.thresholds,
                boundaries=processor.boundaries,
                segment_ids=preds,
                pred_labels_mapped=remapped_preds,
                gt_labels=gt_arr,
                reject_reasons=processor.reject_reasons,
                video_name=f"{dataset_name}-{video_id}"
            )
        
        history["video_id"].append(video_id)
        history["MoF"].append(mof_video)
        history["F1"].append(f1)

    final_mof = global_mof.compute()
    print("\n=== Online ABD — Dataset-level results ===")
    print(f"  Mean MoF : {final_mof*100:.2f}%")
    print(f"  Mean F1  : {np.mean(history['F1'])*100:.2f}%")
    
    return history