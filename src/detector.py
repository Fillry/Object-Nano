import logging
import cv2
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
        
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25, iou_threshold=0.3, device="cpu"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
        self._load_model()

    def _load_model(self):
        """Loads the YOLOv8 model using ultralytics."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {self.model_path} on device {self.device}...")
            self.model = YOLO(self.model_path)
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False, device=self.device)
            logger.info("YOLO model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("Falling back to OpenCV DNN module if model is ONNX...")
            raise e

    def apply_class_agnostic_nms(self, detections, nms_threshold=0.40):
        """
        Applies Class-Agnostic Non-Maximum Suppression.
        Removes overlapping bounding boxes even if they have different class predictions.
        """
        if not detections:
            return []
            
        # Sort detections by confidence score in descending order
        sorted_dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []
        
        for det in sorted_dets:
            bbox = det['bbox']
            overlap = False
            for kept_det in keep:
                iou = calculate_iou(bbox, kept_det['bbox'])
                if iou > nms_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
                
        return keep

    def detect(self, frame):
        """
        Detects vehicles in a frame.
        Returns a list of dictionaries: [{'bbox': [x1, y1, x2, y2], 'class_name': 'car', 'conf': 0.85}, ...]
        """
        if self.model is None:
            return []
            
        # Target classes for vehicles (YOLO COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck)
        vehicle_class_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # We run YOLO with slightly stricter NMS internally
        results = self.model(frame, 
                             conf=self.conf_threshold, 
                             iou=self.iou_threshold, 
                             device=self.device, 
                             verbose=False)
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in vehicle_class_map:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    detections.append({
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        'class_name': vehicle_class_map[cls_id],
                        'conf': conf
                    })
        
        # Apply Class-Agnostic NMS post-processing to eliminate duplicate overlapping boxes
        # (e.g. when YOLO detects a motorcycle and overlapping car, or dual overlapping motorcycle boxes)
        if len(detections) > 1:
            detections = self.apply_class_agnostic_nms(detections, nms_threshold=0.38)
            
        return detections
