#type:ignore
import os
from collections import namedtuple
import csv
import cv2
import model

from dataset import Taco
from model import MaskRCNN
from config import Config
from cv2.typing import MatLike
import cv2
import time
from threading import Thread
from pathlib import Path
from typing import Any # type:ignore[Any]
from queue import Empty, Queue
import os
from random import randint

QUEUE_T = Queue[os.PathLike[Any]]
queue:QUEUE_T = Queue(maxsize=4)

MODEL_DIR = "model_dir"
ROUND = 0 
CLASS_MAP="./taco_config/map_10.csv"

OUT_PATH = Path("out")
IN_PATH = Path("in")

if not OUT_PATH.exists():
    os.makedirs(OUT_PATH)


if not IN_PATH.exists():
    os.makedirs(IN_PATH)
    


def predict(model, dataset, image:MatLike):
    r = model.detect([image], verbose=0)[0]
    class_name:str=""
    max_rect = (-1,-1,-1,-1)
    # Paint the predictions 
    for class_id,rect in zip(r['class_ids'],r['rois']):
        y1, x1, y2, x2 = rect
        rect = (x1,y1,x2-x1,y2-y1)
        if dataset.class_names[class_id].lower() not in ["other"]:
            cv2.rectangle(image, (rect), (0, 255, 0), 2)
            cv2.putText(image, dataset.class_names[class_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if rect>max_rect:
                max_rect = rect
                class_name = dataset.class_names[class_id]
    return class_name


def live_feed(model,dataset,fps:int=30):
    camera =cv2.VideoCapture(0) 
    while True:
        _,image = camera.read()
        predict(model=model,dataset=dataset,image=image)
        cv2.imshow('image', image)
        cv2.waitKey(2)
          


def static(model,dataset,image_path:str)->None:
    if not os.path.exists(image_path):
         raise FileNotFoundError(f"Please Provide a valid file and not")
    image =  cv2.imread(image_path)
    predict(model,dataset,image)
    return image
        

def start_server(queue:QUEUE_T,model,dataset,image:MatLike):
    while True:
        try:
            item = queue.get()
            image = cv2.imread(str(item))
            class_name = predict(model,dataset,image)
            out_path = str(OUT_PATH/item.name)
            cv2.imwrite(out_path,image)
            with open(str(OUT_PATH/f"{item.stem}.txt"),"w") as fp:
                fp.write(class_name)
            print(f"Task Done of , task[{item}]")
            os.remove(item)
        except Empty:
            pass
        except cv2.Error:
            pass

def checker(queue:QUEUE_T):
    directory:os.PathLike[Any]= IN_PATH
    while True:
        for new_item in directory.glob("*"):
            queue.put(new_item)
            time.sleep(0.3)




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
    dataset_test = Taco()
    taco = dataset_test.load_taco(dataset_dir, ROUND, "test", class_map=class_map, return_taco=True)
    dataset_test.prepare()
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
    model_path = str(__import__("pathlib").Path(pretrained_model_path).absolute())
    model.load_weights(model_path, model_path, by_name=True)
    # live_feed(model,dataset_test,30)

    th_server = Thread(target=start_server,args=(queue,model,dataset_text,30))
    th_checker = Thread(target=checker,args=(queue,))

    th_server.start()
    th_checker.start()

    th_server.join()
    th_checker.join()
    
    #DEBUG
    '''
    start = time.time()
    for i in range(1,10):
        static(model,dataset_test,"/home/kali_37/Documents/prg_lang/python/ML/TACO2/detector/image.png")
        print("Time taken: ", time.time()-start)
    '''