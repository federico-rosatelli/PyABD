import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import torch
from src.core import abd, refinement
from src.utils.data_loader import BreakfastDataset
from src.utils.visualization import plot_abd_results
from src.utils import config
from src.metrics import accuracy
from scipy.ndimage import zoom

def main():
    br_conf = config.getConfigYAML("config/breakfast.yaml")
    bn_conf = config.getConfigYAML("config/boundaries.yaml")
    logger  = config.getLogger("development")
    
    bf_dataset = BreakfastDataset(**br_conf)
    dataset = bf_dataset.getDataset()
    K = br_conf['thrashold_classes']
    for item in dataset:
        time_s = time.time()

        features = item['video_feature']
        video_label = item['video_label']
        video_id = item['video_id']
        #print(list(set(video_label)))
        n_labels = list(set(video_label))
        video_feature = torch.tensor(features, dtype=torch.float32)
        
        boundaries, similarity = abd.detect_boundaries(video_feature, **bn_conf)
        #print(boundaries.shape[0],similarity.shape[0])
        
        similarity = torch.cat([similarity, similarity[-1].unsqueeze(0)])
        logger.info(f"Boundaries: {len(boundaries)}. Time: {(time.time()-time_s):.2f}")

        if len(boundaries) > K:
            pred = refinement.refine_segments(video_feature, boundaries, K)

            target_len = len(video_feature)

            if len(pred) != target_len:
                zoom_factor = target_len / len(pred)
                pred = zoom(pred, zoom_factor, order=0)

            gt_arr = np.array(video_label)
            if len(gt_arr) != target_len:
                zoom_factor = target_len / len(gt_arr)
                gt_resized = zoom(gt_arr, zoom_factor, order=0)
            else:
                gt_resized = gt_arr

            preds_cut = pred 
            gt_cut = gt_resized

            mapped_preds = accuracy.calculate_hungerian_mapping(preds_cut, gt_cut)
            mof = accuracy.calculate_mof(mapped_preds, gt_cut)
            f1 = accuracy.calculate_f1(mapped_preds,gt_cut)

            

            logger.info(f"Video {video_id} - Accuracy (MoF): {mof*100:.2f}% - Accuracy (F1): {f1*100:.2f}%")
            plot_abd_results(
                similarity=similarity,
                boundaries=boundaries,
                pred_labels_mapped=mapped_preds,
                gt_labels=gt_cut,
                video_name=video_id
            )




if __name__ == "__main__":
    main()