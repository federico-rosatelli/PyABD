import csv
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


def run_offline_mode(dataset, dataset_name:str ,alpha:float, K:int):

    tq = tqdm(dataset, desc=f"Processing dataset {dataset_name}", unit="video")

    history = {
        "video_id": [],
        "boundaries": [],
        "MoF": [],
        "F1": []
    }

    global_mof = GlobalMoF()

    alpha = alpha if alpha > 0 and alpha <= 1 else 0.6

    times = []

    for item in tq:
        time_s = time.time()

        features = item['video_feature']
        video_label = item['video_label']
        video_id = item['video_id']
        
        video_feature = torch.tensor(features, dtype=torch.float32)
        target_len = video_feature.shape[0]

        dynamic_size = int(alpha * (target_len / K))
        
        if dynamic_size % 2 == 0:
            dynamic_size += 1
            
        dynamic_size = max(3, dynamic_size)
        
        boundaries, similarity = abd.detect_boundaries(video_feature, dynamic_size, dynamic_size)
        
        similarity = torch.cat([similarity, similarity[-1].unsqueeze(0)])

        # k_log = dynamic_K(similarity, K)
        
        k_def = K #+ k_log

        

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
        times.append(time.time() - time_s)

        
        tq.set_postfix_str(f"MoF: {mof_video*100:.2f}% - F1: {f1*100:.2f}%")
        if f1 > 0.85:
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
    mean_f1 = np.mean(history['F1'])
    # print("\n=== Offline ABD — Dataset-level results ===")
    # print(f"  Mean MoF : {final_mof*100:.2f}%")
    # print(f"  Mean F1  : {np.mean(history['F1'])*100:.2f}%")
    
    return final_mof, mean_f1, sum(times)/len(times)
        


def run_grid_search(dataset_name: str, boundaries_type: str, alphas: list, Ks: list, output_csv: str = "grid_search_results.csv"):

    dataset_conf = config.getConfigYAML("config/dataset.yaml")
    
    if dataset_name == "breakfast":
        br_conf = dataset_conf["datasets"]['breakfast']
        dataset = BreakfastDataset(**br_conf).getDataset()
    elif dataset_name == "50salads":
        br_conf = dataset_conf["datasets"]['50salads']
        dataset = SaladsDataset(br_conf['name'], br_conf['dataset_path'], br_conf["boundaries"][boundaries_type]['threshold_classes'])
    else:
        raise ValueError("Error Loading Dataset")
        
        
    total_iters = len(alphas) * len(Ks)
    print(f"Running Grid Search for Dataset: {dataset_name}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Alpha", "K", "Mean_MoF", "Mean_F1", "Mean_Time"])
        
        with tqdm(total=total_iters, desc="Grid Search Progress") as pbar:
            for alpha in alphas:
                for K in Ks:
                    pbar.set_postfix({"Alpha": alpha, "K": K})

                    mof, f1, times = run_offline_mode(dataset, dataset_name, alpha, K)
                    
                    writer.writerow([dataset_name, alpha, K, f"{mof*100:.2f}", f"{f1*100:.2f}", f"{times:.2f}"])
                    f.flush()
                    
                    pbar.update(1)




def dynamic_K(similarity: torch.Tensor, K: int):
    b_log = math.log(len(similarity)+1)
    
    k_log = (b_log - K)
    
    if isinstance(similarity, torch.Tensor):
        M_v, m_v, me_v = similarity.max(), similarity.min(), similarity.mean()
    else:
        M_v, m_v, me_v = max(similarity), min(similarity), np.mean(similarity)

    x = (M_v - m_v) / (me_v - m_v) if me_v != m_v else 0

    return int(k_log*x)

