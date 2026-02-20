from datasets import load_dataset
import kagglehub
from glob import glob
import os

import numpy as np
from tqdm import tqdm

class BreakfastDataset:

    def __init__(self,
                 name:str,
                 dataset:str, 
                 dataset_path:str, 
                 output_path:str, 
                 split_type:str, 
                 split_name:str, 
                 streaming:bool, 
                 mode:str, 
                 writer_batch_size:int=1):
        
        self.name = name
        self.dataset_name = dataset
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.split_type = split_type
        self.split_name = split_name
        self.writer_batch_size = writer_batch_size
        self.streaming = streaming
        self.mode = mode
        self.dataset = self._load_br_dataset()
        pass

    def _load_br_dataset(self):
        try:
            dataset = load_dataset(self.dataset_name,
                                   name=self.split_name,
                                   cache_dir=self.dataset_path,
                                   writer_batch_size=self.writer_batch_size,
                                   streaming=self.streaming
                                   )
        except Exception as e:
            print(f"Error while Loading Breakfast Dataset: {e}")
            return
        
        return dataset[self.split_type]

    def getDataset(self):
        return self.dataset
    
    def info(self):
        return f"Breakfast Dataset: {self.name}, Split: {self.split_type}, Streaming: {self.streaming}"



class SaladsDataset:
    def __init__(self,name, dataset_path, sample_rate):
        
        self.name = name
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.action_dict = self._get_action_dict()
        self.features, self.labels, self.video_ids = self._load_salads_dataset()
        pass

    def _get_action_dict(self):
        mapping_file = os.path.join(self.dataset_path, 'mapping', 'mappingeval.txt')
        action_dict = {}
        with open(mapping_file, 'r') as f:
            for line in f.readlines():
                idx, action_name = line.strip().split() 
                action_dict[action_name] = int(idx)
        return action_dict

    def _load_salads_dataset(self):
        features = []
        labels = []

        features_path = os.path.join(self.dataset_path, "features/rgb")
        labels_path = os.path.join(self.dataset_path, "groundTruth")

        videos_path = glob(os.path.join(labels_path, "*"))
        video_ids = [os.path.basename(video) for video in videos_path]

        for video_id in tqdm(video_ids, desc="Extracting features"):
            
            feature_path = os.path.join(features_path, video_id + ".txt")
            label_path = os.path.join(labels_path, video_id)
            if os.path.exists(feature_path) and os.path.exists(label_path):
                features.append(np.loadtxt(feature_path))
                
                with open(label_path, 'r') as f:
                    content = f.read().split('\n')[:-1]
                    classes = np.zeros(len(content), dtype=np.int64)
                    for i in range(len(content)):
                        classes[i] = self.action_dict[content[i]]

                labels.append(classes)
            else:
                print(f"Warning: Missing features or labels for video {video_id}")
        
        return features, labels, video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        return {
            "video_feature": self.features[idx],
            "video_label": self.labels[idx],
            "video_id": self.video_ids[idx]
        }