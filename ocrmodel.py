import os
import sys
import numpy as np
import time
from PIL import Image

# 1. Setup Environment Path
project_root = os.path.dirname(os.path.abspath(__file__))
uvdoc_path = os.path.join(project_root, "uvdoc")
# Add uvdoc folder so uvdoc's internal 'import utils' works
if uvdoc_path not in sys.path:
    sys.path.append(uvdoc_path)

from uvdoc.unwarp import unwarp_img

# Third-party imports
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddlex import create_model

class OCRPipeline:
    def __init__(self, device="cuda:0", debug=False):
        """
        Initializes the three-stage OCR pipeline using local modules and 3rd party models.
        """
        self.device = device
        self.debug = debug
        # Resolve absolute path for checkpoint to avoid 'File Not Found' errors
        self.uvdoc_ckpt = os.path.join(project_root, "uvdoc/model/best_model.pkl")
        
        # 2. Initialize VietOCR (Recognition)
        print("Initializing VietOCR...")
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['cnn']['pretrained'] = False
        config['device'] = self.device
        self.recognizer = Predictor(config)

        # 3. Initialize PaddleX (Detection)
        print("Initializing PaddleX Detector...")
        self.detector = create_model("PP-OCRv5_server_det")

    def _get_bounding_rects(self, det_output):
        """
        Converts PaddleX polygons to sorted bounding rectangles.
        """
        rects = []
        for res in det_output:
            
            # For eval purpose
            if self.debug:
                res.save_to_img(save_path="./output/")
                res.save_to_json(save_path="./output/")
            
            polys = res['dt_polys']
            for poly in polys:
                poly_np = np.array(poly)
                x_min, y_min = poly_np.min(axis=0)
                x_max, y_max = poly_np.max(axis=0)
                rects.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        
        # Sort by Y-coordinate for top-to-bottom reading order
        return sorted(rects, key=lambda x: x[1])

    def predict(self, img_path):
        """
        Full pipeline: Unwarp (Local) -> Detect (PaddleX) -> Recognize (VietOCR).
        """
        # Step 1: Unwarp the document using the local uvdoc module
        unwarp_img(img_path, ckpt_path=self.uvdoc_ckpt)
        unwarped_path = os.path.splitext(img_path)[0] + "_unwarp.png"
        
        # Step 2: Detect text
        det_output = self.detector.predict(input=unwarped_path)
        
        # Step 3: Process coordinates
        bounding_boxes = self._get_bounding_rects(det_output)
        
        # Step 4: Recognize text line by line
        results = []
        with Image.open(unwarped_path) as img:
            for box in bounding_boxes:
                cropped_line = img.crop(box)
                text = self.recognizer.predict(cropped_line)
                results.append(text)
        
        # Clean up the temporary unwarped image created by UVDoc
        if not self.debug and os.path.exists(unwarped_path):
            os.remove(unwarped_path)
        
        return results

# Example Usage:
if __name__ == "__main__":
    pipeline = OCRPipeline()
    
    test_image = "test_img/vnese_test1.jpg"
    if os.path.exists(test_image):
        start_time = time.time()
        text_lines = pipeline.predict(test_image)
        end_time = time.time()
        
        print(f"Finished in {end_time - start_time:.2f} seconds")
        print("\n".join(text_lines))
    else:
        print(f"Error: Could not find test image at {test_image}")