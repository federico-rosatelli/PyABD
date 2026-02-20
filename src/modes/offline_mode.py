from tqdm import tqdm
from src.utils import config
from src.utils.data_loader import BreakfastDataset, SaladsDataset
from src.core import abd, refinement
from src.metrics import accuracy
from scipy.ndimage import zoom
import torch
import numpy as np
from src.utils.visualization import plot_abd_results
import time


def run_offline_mode(dataset_name:str, log=False):

    dataset_conf = config.getConfigYAML("config/dataset.yaml")
    if dataset_name == "breakfast":
        br_conf = dataset_conf["datasets"]['breakfast']
        dataset = BreakfastDataset(**br_conf).getDataset()
    elif dataset_name == "50salads":
        br_conf = dataset_conf["datasets"]['50salads']
        dataset = SaladsDataset(br_conf['name'], br_conf['dataset_path'], 1)
    
    bn_conf = config.getConfigYAML("config/boundaries.yaml")

    if log:
        logger  = config.getLogger("production")
    K = bn_conf['thrashold_classes']
    print(f"Running offline mode on {dataset_name} dataset with K={K} classes...")
    tq = tqdm(dataset, desc=f"Processing dataset {dataset_name}", unit="video")

    history = {
        "video_id": [],
        "MoF": [],
        "F1": []
    }

    for item in tq:
        time_s = time.time()

        features = item['video_feature']
        video_label = item['video_label']
        video_id = item['video_id']
        
        video_feature = torch.tensor(features, dtype=torch.float32)
        target_len = video_feature.shape[0]
        
        boundaries, similarity = abd.detect_boundaries(video_feature, bn_conf["kernel_size"], bn_conf["window_size"])
        
        similarity = torch.cat([similarity, similarity[-1].unsqueeze(0)])

        if len(boundaries) == 0:
            pred = torch.zeros(target_len, dtype=torch.long)
        elif len(boundaries) <= K:
            pred = refinement.refine_segments(video_feature, boundaries, len(boundaries))
        else:
            pred = refinement.refine_segments(video_feature, boundaries, K)

        

        preds_cut = pred.numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
        gt_arr  = np.array(video_label)

        if len(preds_cut) != target_len:
            preds_cut = zoom(preds_cut, target_len / len(preds_cut), order=0)
        if len(gt_arr) != target_len:
            gt_arr = zoom(gt_arr, target_len / len(gt_arr), order=0)


        mapped_preds = accuracy.calculate_hungerian_mapping(preds_cut, gt_arr)
        mof = accuracy.calculate_mof(mapped_preds, gt_arr)
        f1 = accuracy.calculate_f1(mapped_preds,gt_arr)

        
        if log:
            logger.info(f"Video {video_id} | Accuracy (MoF): {mof*100:.2f}% | Accuracy (F1): {f1*100:.2f}% | Time: {(time.time()-time_s):.2f}s")
        tq.set_postfix_str(f"MoF: {mof*100:.2f}% - F1: {f1*100:.2f}%")
        if f1 > 0.4:
            plot_abd_results(
                similarity=similarity,
                boundaries=boundaries,
                pred_labels_mapped=mapped_preds,
                gt_labels=gt_arr,
                video_name=f"{dataset_name}-{video_id}"
            )
        
        history["video_id"].append(video_id)
        history["MoF"].append(mof)
        history["F1"].append(f1)
    
    return history
        

