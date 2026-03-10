import math
from tqdm import tqdm
from src.utils import config
from src.utils.data_loader import BreakfastDataset, SaladsDataset
from src.core import abd, refinement
from src.metrics.accuracy import calculate_mof, GlobalMoF, calculate_f1
from scipy.ndimage import zoom
import torch
import numpy as np
from src.utils.visualization import plot_abd_results
import time


def run_offline_mode(dataset_name:str, boundaries_type:str, log=False):

    dataset_conf = config.getConfigYAML("config/dataset.yaml")
    if dataset_name == "breakfast":
        br_conf = dataset_conf["datasets"]['breakfast']
        dataset = BreakfastDataset(**br_conf).getDataset()
    elif dataset_name == "50salads":
        br_conf = dataset_conf["datasets"]['50salads']
        dataset = SaladsDataset(br_conf['name'], br_conf['dataset_path'], br_conf["boundaries"][boundaries_type]['threshold_classes'])
    
    #bn_conf = config.getConfigYAML("config/boundaries.yaml")

    kernel_size = br_conf["boundaries"][boundaries_type]['kernel_size']
    window_size = br_conf["boundaries"][boundaries_type]['window_size']


    if log:
        logger  = config.getLogger("production")
    K = dataset_conf["datasets"][dataset_name]["boundaries"][boundaries_type]['threshold_classes']
    print(f"Running offline mode on {dataset_name} dataset with K={K} classes...")
    tq = tqdm(dataset, desc=f"Processing dataset {dataset_name}", unit="video")

    history = {
        "video_id": [],
        "boundaries": [],
        "MoF": [],
        "F1": []
    }

    global_mof = GlobalMoF()

    for item in tq:
        time_s = time.time()

        features = item['video_feature']
        video_label = item['video_label']
        video_id = item['video_id']
        
        video_feature = torch.tensor(features, dtype=torch.float32)
        target_len = video_feature.shape[0]
        
        boundaries, similarity = abd.detect_boundaries(video_feature, kernel_size, window_size)
        
        similarity = torch.cat([similarity, similarity[-1].unsqueeze(0)])
        
        k_log = int(math.log(len(boundaries)+1))

        k_log = (k_log//2 * 1) if k_log >= K//2 else (k_log//2 * -1)
        
        k_def = K + k_log

        

        if len(boundaries) == 0:
            pred = torch.zeros(target_len, dtype=torch.long)
        elif len(boundaries) <= k_def:
            pred = refinement.refine_segments(video_feature, boundaries, len(boundaries))
        else:
            pred = refinement.refine_segments(video_feature, boundaries, k_def)

        

        preds_cut = pred.numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
        gt_arr  = np.array(video_label)

        if len(preds_cut) != target_len:
            preds_cut = zoom(preds_cut, target_len / len(preds_cut), order=0)
        if len(gt_arr) != target_len:
            gt_arr = zoom(gt_arr, target_len / len(gt_arr), order=0)


        mof_video, remapped_preds = calculate_mof(preds_cut, gt_arr)
        global_mof.update(remapped_preds, gt_arr)
        f1 = calculate_f1(remapped_preds, gt_arr)

        
        if log:
            logger.info(f"Video {video_id} | Accuracy (MoF): {mof_video*100:.2f}% | Accuracy (F1): {f1*100:.2f}% | Time: {(time.time()-time_s):.2f}s")
        tq.set_postfix_str(f"MoF: {mof_video*100:.2f}% - F1: {f1*100:.2f}%")
        if f1 > 0.4:
            plot_abd_results(
                similarity=similarity,
                boundaries=boundaries,
                pred_labels_mapped=remapped_preds,
                gt_labels=gt_arr,
                video_name=f"{dataset_name}-{video_id}"
            )
        
        history["video_id"].append(video_id)
        history["MoF"].append(mof_video)
        history["F1"].append(f1)

    final_mof = global_mof.compute()
    print("\n=== Offline ABD — Dataset-level results ===")
    print(f"  Mean MoF : {final_mof*100:.2f}%")
    print(f"  Mean F1  : {np.mean(history['F1'])*100:.2f}%")
    
    return history
        

