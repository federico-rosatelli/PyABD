from datasets import load_dataset

class BreakfastDataset:

    def __init__(self, 
                 dataset:str, 
                 dataset_path:str, 
                 output_path:str, 
                 split_type:str, 
                 split_name:str, 
                 streaming:bool, 
                 mode:str, 
                 writer_batch_size:int=1):
        
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
