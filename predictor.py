import os
import csv
import cv2
import model
from typing import Any #type:ignore[Any]
from dataset import Taco
from model import MaskRCNN
from config import Config
from cv2.typing import MatLike
from pathlib import Path
from pydantic import BaseModel

MODEL_DIR = "model_dir"
ROUND = 0 
CLASS_MAP="./taco_config/map_10.csv"



class Detected(BaseModel):
    class_ids:list[str] = []
    rois:list[tuple[int,int,int,int]] = []


class Dataset(Taco):
    class_names:dict[str,str] = {}


    def nload_taco(self, dataset_dir:str, round:int, subset:str, class_map:dict[Any,Any]|None=None):
        super().load_taco(dataset_dir, round, subset,class_map=class_map) # type:ignore[UnknownMemberType]
        
    def nprepare(self):
        self.prepare() # type:ignore[UnknownMemberType]



def predict(model:MaskRCNN, dataset:Dataset, image:MatLike):
    r:Detected = Detected(**model.detect([image], verbose=0)[0])
    # Paint the predictions 
    for class_id,rect in zip(r.class_ids,r.rois):
        y1, x1, y2, x2 = rect
        _ = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _= cv2.putText(image, dataset.class_names[class_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)



def live_feed(model:MaskRCNN, dataset:Dataset,fps:int=30):
    camera =cv2.VideoCapture(0) 
    while True:
        _,image = camera.read()
        predict(model=model,dataset=dataset,image=image)
        cv2.imshow('image', image)
        _= cv2.waitKey(2)
          


def static(model:MaskRCNN, dataset:Dataset,image_path:str)->None:
    if not os.path.exists(image_path):
         raise FileNotFoundError(f"Please Provide a valid file and not")
    image =  cv2.imread(image_path)
    predict(model,dataset,image)
    cv2.imshow('image', image)
    _= cv2.waitKey(1)
        


if __name__ == '__main__':
    pretrained_model_path = "epoch_565.h5"
    dataset_dir="../data"

    # Read map of target classes
    class_map = {}
    map_to_one_class = {}
    with open(CLASS_MAP) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}
        map_to_one_class = {c: 'Litter' for c in class_map}

    
    # Test dataset
    dataset_test = Dataset(Taco())
    dataset_test.nload_taco(dataset_dir, ROUND, "test", class_map=class_map)
    dataset_test.nprepare()
    nr_classes = dataset_test.num_classes

    class TacoTestConfig(Config):
            NAME = "taco"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 10
            NUM_CLASSES = nr_classes
            USE_OBJECT_ZOOM = False
    config = TacoTestConfig()
    config.display()
    model = MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model_path = Path(pretrained_model_path).absolute()
    
    model.load_weights(model_path, model_path, by_name=True)
    

    live_feed(model,dataset_test,30)
    
    #DEBUG
    '''
    start = time.time()
    for i in range(1,10):
        static(model,dataset_test,"/home/kali_37/Documents/prg_lang/python/ML/TACO2/detector/image.png")
        print("Time taken: ", time.time()-start)
    '''