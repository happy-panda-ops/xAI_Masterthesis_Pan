import os
HOME = os.getcwd()

!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from otsu_filter import otsu_filter as otsu

class ObjectDetection:
    def __init__(self, model_name="YOLOv5s"):
        self.model = ultralytics.YOLO(model_name)
    
    # choose target area from simplified and transformed polygon area
    def get_area(self, input_path, target_polygon)
        #TODO: implement area selection
        return target_area

    # detection on target area
    def get_prediction(self, target_area):
        # FIXME: check if target_area is a valid input
        results = self.model(target_area)
        return results
    
    # reduce the BBOX using kNN
    def reduce_knn(self, results):
        #TODO: implement kNN using feature extraction
        #TODO: samples -> features using resNet18 -> kNN
        return reduced_bbox
    
    # reduce the size of BBOX using otsu filter
    def reduce_otsu(self, results):
        # FIXME: check if otsu rightly applied
        otsu_filter = otsu()
        reduced_bbox = otsu_filter.apply_otsu_filter(results)
        return reduced_bbox
    
    