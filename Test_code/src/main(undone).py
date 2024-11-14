from segmentation import ObjectSegmentation as getSeg
from transformation import PolygonTransformer as getTrans
from detection import ObjectDetection as getDet

class main:
    
    def __init__(self):
        self.input_img = input_img
        self.output_folder = output_folder
        self.seg_model_path = seg_model_path
        self.segmentation_json_path = segmentation_json_path
        
    def run_segmentation(self, seg_model_path, input_img, output_folder):
        seg_model_path = seg_model_path
        seg_input_image = input_img
        seg_output_folder = output_folder
        segmenter = getSeg(seg_model_path, seg_input_image, seg_output_folder, model_type='yolo')
        segmenter.run()
        
    def run_transformation(self, input_img, segmentation_json_path, output_folder):
        transformer = getTrans(input_img, segmentation_json_path, output_folder)
        transformer.run()
        
    def run_detection(self, det_model_path, input_img, output_folder):
        detector = getDet(det_model_path, input_img, output_folder)
        detector.run()

if __name__ == "__main__":
    seg_model_path = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/src/models/seg/YOLO-seg-test-3best.pt"
    
    det_model_path = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/src/models/detec/Mixed_YOLO_d1+d2+s1_test_1-best.pt"
    
    #! captured image
    input_img = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/example/demo_tests/1_seg/Model_2_0004.jpeg"

    output_folder = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/output/Model_2_0004"
    
    
    #! name of input_image end with "_seg.json" 
    segmentation_json_path = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/output/Model_2_0004/Model_2_0004_seg.json"

    #! name of input_image end with "_tran_transformed.png" 
    det_input = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/output/Model_2_0004/Model_2_0004_tran_transformed.png"
    
    main().run_segmentation(seg_model_path, input_img, output_folder)
    main().run_transformation(input_img, segmentation_json_path, output_folder)
    main().run_detection(det_model_path, det_input, output_folder)