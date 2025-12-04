import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import yaml
import datetime
import logging.config
import torch
from src.core import abd,refinement
from src.utils.data_loader import BreakfastDataset
from src.utils.visualization import plot_abd_results
from src.metrics.accuracy import calculate_mof

def main():
    br_conf = getConfigYAML("config/breakfast.yaml")
    bn_conf = getConfigYAML("config/boundaries.yaml")
    logger = getLogger("development")
    
    bf_dataset = BreakfastDataset(**br_conf)
    dataset = bf_dataset.getDataset()
    K = 6
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

            min_len = min(len(pred), len(video_label))
            preds_cut = pred[:min_len]
            gt_cut = video_label[:min_len]

            
            mof, remapped_preds = calculate_mof(preds_cut, gt_cut)

            logger.info(f"Video {video_id} - Accuracy (MoF): {mof:.4f} ({mof*100:.2f}%)")
            plot_abd_results(
                similarity=similarity,
                boundaries=boundaries,
                pred_labels_mapped=remapped_preds,
                gt_labels=video_label,
                video_name=video_id
            )


            



def getConfigYAML(conf_file:str) -> any:
    with open(conf_file, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

def getLogger(name:str) -> logging.Logger:
    
    config = getConfigYAML("config/logger.yaml")
    if name == "staging" or name == "production":
        config['handlers']['file']['filename'] = "logs/"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".log"
    else:
        config['handlers']['file']['filename'] = "logs/dev.log"
    logging.config.dictConfig(config)

    logger = logging.getLogger(name)
    return logger


if __name__ == "__main__":
    main()