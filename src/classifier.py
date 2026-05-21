import cv2
import numpy as np
import math
import logging
import os
import json

logger = logging.getLogger("ITS_Classifier")

# Custom Color Palette Mapping (Reference colors in BGR format)
# Note: OpenCV cv2.cvtColor expects BGR input
COLOR_PALETTE_BGR = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128),
    "Metallic Green": (111, 124, 74),  # HEX #4A7C6F -> RGB (74, 124, 111) -> BGR (111, 124, 74)
    "Chartreuse": (50, 215, 180),     # HEX #B4D732 -> RGB (180, 215, 50) -> BGR (50, 215, 180)
    "Blue": (235, 206, 135),          # HEX #87CEEB -> RGB (135, 206, 235) -> BGR (235, 206, 135)
    "Charcoal": (79, 69, 54),         # HEX #36454F -> RGB (54, 69, 79) -> BGR (79, 69, 54)
    "Silver": (192, 192, 192),
    "Gold": (55, 175, 212),           # HEX #D4AF37 -> RGB (212, 175, 55) -> BGR (55, 175, 212)
    "Navy Blue": (128, 0, 0),         # HEX #000080 -> RGB (0, 0, 128) -> BGR (128, 0, 0)
    "Slate Blue": (144, 128, 112),    # HEX #708090 -> RGB (112, 128, 144) -> BGR (144, 128, 112)
    "Bronze": (83, 120, 140),         # HEX #8C7853 -> RGB (140, 120, 83) -> BGR (83, 120, 140)
    "Red": (60, 20, 220),             # HEX #DC143C -> RGB (220, 20, 60) -> BGR (60, 20, 220)
    "Maroon": (0, 0, 128),            # HEX #800000 -> RGB (128, 0, 0) -> BGR (0, 0, 128)
    "Pink": (193, 182, 255),          # HEX #FFB6C1 -> RGB (255, 182, 193) -> BGR (193, 182, 255)
    "Bronze Gold": (63, 142, 175),    # HEX #AF8E3F -> RGB (175, 142, 63) -> BGR (63, 142, 175)
    "Bronze Gray": (108, 120, 130),   # HEX #82786C -> RGB (130, 120, 108) -> BGR (108, 120, 130)
    "Bronze Silver": (154, 163, 169), # HEX #A9A39A -> RGB (169, 163, 154) -> BGR (154, 163, 169)
    "Orange": (0, 140, 255),          # HEX #FF8C00 -> RGB (255, 140, 0) -> BGR (0, 140, 255)
    "Yellow": (0, 230, 255),          # HEX #FFE600 -> RGB (255, 230, 0) -> BGR (0, 230, 255)
    "Green": (0, 128, 0),
    "Light Green": (144, 238, 144),   # HEX #90EE90 -> RGB (144, 238, 144) -> BGR (144, 238, 144)
    "Dark Green": (0, 100, 0),        # HEX #006400 -> RGB (0, 100, 0) -> BGR (0, 100, 0)
    "Olive Green": (35, 142, 107)     # HEX #6B8E23 -> RGB (107, 142, 35) -> BGR (35, 142, 107)
}

# Pre-convert palette reference colors to CIELab to optimize inference speed
COLOR_PALETTE_LAB = {}
for name, bgr in COLOR_PALETTE_BGR.items():
    # Convert BGR to CIELab (OpenCV requires 3D float array for precise conversion)
    bgr_img = np.uint8([[bgr]])
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    COLOR_PALETTE_LAB[name] = lab_img[0][0].astype(float)

def bgr_to_lab(bgr_color):
    """Converts a BGR color tuple to CIELab."""
    bgr_img = np.uint8([[bgr_color]])
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    return lab_img[0][0].astype(float)

def delta_e_distance(lab1, lab2):
    """
    Computes a light-weight Delta E style perceptual distance in CIELab space.
    We apply a lower weight to the L (Luminance) channel to reduce sensitivity to shadows/highlights.
    """
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    # L weight = 0.35 (makes color classification robust to shadows on cars)
    return math.sqrt(0.35 * (dL ** 2) + (da ** 2) + (db ** 2))

def match_closest_color(bgr_color):
    """Matches a BGR color to the closest color name in the custom palette using CIELab Delta E."""
    lab_color = bgr_to_lab(bgr_color)
    min_dist = float('inf')
    best_match = "Unknown"
    
    for name, ref_lab in COLOR_PALETTE_LAB.items():
        dist = delta_e_distance(lab_color, ref_lab)
        if dist < min_dist:
            min_dist = dist
            best_match = name
            
    return best_match

class VehicleClassifier:
    def __init__(self, config=None):
        """
        config: Classifiers configuration dictionary
        """
        logger.info("Vehicle Classifier initialized using K-Means & CIELab Delta E Color Palette.")
        self.config = config
        self.brand_enabled = False
        self.brand_backend = None
        
        if config and 'brand' in config:
            brand_cfg = config['brand']
            if brand_cfg.get('enabled', False):
                self.brand_model_path = brand_cfg.get('model_path', 'models/brand_classifier.pth')
                self.brand_classes = brand_cfg.get('classes', [])
                self.brand_conf_threshold = brand_cfg.get('conf_threshold', 0.50)
                self.brand_enabled = True
                self._load_brand_model()

    def _load_brand_model(self):
        if not os.path.exists(self.brand_model_path):
            logger.warning(f"Brand classifier model not found at {self.brand_model_path}. Fallback to heuristics.")
            self.brand_enabled = False
            return

        ext = os.path.splitext(self.brand_model_path)[1].lower()
        if ext in ['.pth', '.pt']:
            try:
                import torch
                import sys
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                tinyvit_path = os.path.join(script_dir, "TinyViT")
                if tinyvit_path not in sys.path:
                    sys.path.append(tinyvit_path)
                from models.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
                
                logger.info(f"Loading PyTorch Brand Classifier from {self.brand_model_path}")
                checkpoint = torch.load(self.brand_model_path, map_location="cpu")
                
                # Dynamic model configuration matching checkpoint
                state_dict = checkpoint['model_state_dict']
                head_weight_shape = state_dict['head.weight'].shape
                in_features = head_weight_shape[1]
                num_classes = head_weight_shape[0]
                
                if in_features == 320:
                    self.pytorch_model = tiny_vit_5m_224(pretrained=False)
                elif in_features == 448:
                    self.pytorch_model = tiny_vit_11m_224(pretrained=False)
                elif in_features == 576:
                    self.pytorch_model = tiny_vit_21m_224(pretrained=False)
                else:
                    logger.warning(f"Unexpected in_features={in_features}. Defaulting to tiny_vit_11m_224.")
                    self.pytorch_model = tiny_vit_11m_224(pretrained=False)
                
                self.pytorch_model.head = torch.nn.Linear(self.pytorch_model.head.in_features, num_classes)
                self.pytorch_model.load_state_dict(state_dict)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.pytorch_model.to(self.device)
                self.pytorch_model.eval()
                self.brand_backend = 'pytorch'
                
                # Update classes mapping from checkpoint
                if 'classes' in checkpoint:
                    self.brand_classes = checkpoint['classes']
                logger.info(f"PyTorch Brand Classifier loaded successfully. Classes: {self.brand_classes}")
            except Exception as e:
                logger.error(f"Failed to load PyTorch brand classifier: {e}")
                self.brand_enabled = False

        elif ext == '.onnx':
            try:
                import onnxruntime as ort
                logger.info(f"Loading ONNX Brand Classifier from {self.brand_model_path}")
                
                # Check for companion classes JSON file
                classes_path = os.path.join(os.path.dirname(self.brand_model_path), "brand_classes.json")
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        self.brand_classes = json.load(f)
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(self.brand_model_path, providers=providers)
                self.brand_backend = 'onnx'
                logger.info(f"ONNX Brand Classifier loaded successfully. Classes: {self.brand_classes}")
            except Exception as e:
                logger.error(f"Failed to load ONNX brand classifier: {e}")
                self.brand_enabled = False

        elif ext in ['.engine', '.trt']:
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit
                
                logger.info(f"Loading TensorRT Brand Classifier from {self.brand_model_path}")
                
                classes_path = os.path.join(os.path.dirname(self.brand_model_path), "brand_classes.json")
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        self.brand_classes = json.load(f)

                self.trt_logger = trt.Logger(trt.Logger.WARNING)
                with open(self.brand_model_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                    self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                
                self.trt_context = self.trt_engine.create_execution_context()
                
                # Setup bindings / buffers
                self.trt_inputs = []
                self.trt_outputs = []
                self.trt_allocations = []
                
                for i in range(self.trt_engine.num_bindings):
                    name = self.trt_engine.get_binding_name(i)
                    is_input = self.trt_engine.binding_is_input(i)
                    dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
                    shape = self.trt_engine.get_binding_shape(i)
                    size = np.prod(shape)
                    nbytes = size * np.dtype(dtype).itemsize
                    
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    dev_mem = cuda.mem_alloc(nbytes)
                    self.trt_allocations.append(int(dev_mem))
                    
                    binding_info = {
                        'name': name, 'dtype': dtype, 'shape': shape,
                        'size': size, 'host_mem': host_mem, 'dev_mem': dev_mem
                    }
                    if is_input:
                        self.trt_inputs.append(binding_info)
                    else:
                        self.trt_outputs.append(binding_info)
                
                self.brand_backend = 'tensorrt'
                logger.info(f"TensorRT Brand Classifier loaded successfully. Classes: {self.brand_classes}")
            except Exception as e:
                logger.error(f"Failed to load TensorRT brand classifier: {e}")
                self.brand_enabled = False

    def _preprocess_image(self, crop):
        # Resize to 224x224
        img = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension (1, 3, 224, 224)
        img = np.expand_dims(img, axis=0)
        return img


    def classify_color(self, crop, k=3, min_proportion=0.28):
        """
        Extracts dominant colors using K-Means on the hood/body area of the vehicle.
        Matches color clusters to custom color palette using CIELab Delta E distance.
        Supports specific Two-Tone mappings:
          - Blue + White -> Blue-White
          - Red + White -> Red-White
          - Yellow + Green -> Yellow-Green
        """
        if crop is None or crop.size == 0:
            return "Unknown"

        h, w, _ = crop.shape
        
        # Focus on the grille/hood area: center height (35% to 75%), center width (20% to 80%)
        # This isolates the car body and avoids dark tires, ground shadow, or windshield reflections.
        y1, y2 = int(h * 0.35), int(h * 0.75)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        roi = crop[y1:y2, x1:x2]
        
        if roi.size == 0:
            roi = crop

        # Reshape for OpenCV K-Means
        pixels = roi.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Define criteria and run K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        try:
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
        except Exception as e:
            logger.error(f"K-Means error: {e}")
            return "Unknown"

        # Calculate proportions of each color cluster
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        proportions = counts / total_pixels
        
        # Sort indices by proportion in descending order
        sorted_indices = np.argsort(proportions)[::-1]
        
        detected_color_names = []
        for idx in sorted_indices:
            prop = proportions[idx]
            bgr_center = centers[idx].astype(int)
            color_name = match_closest_color(bgr_center)
            
            # Record significant colors that represent a high proportion of the surface area
            if prop >= min_proportion and color_name not in detected_color_names:
                detected_color_names.append(color_name)

        # Apply custom Two-Tone mapping rules
        if len(detected_color_names) >= 2:
            primary = detected_color_names[0]
            secondary = detected_color_names[1]
            
            # Map combinations to requested two-tone names
            two_tone_pairs = {
                frozenset(["Blue", "White"]): "Blue-White",
                frozenset(["Red", "White"]): "Red-White",
                frozenset(["Yellow", "Green"]): "Yellow-Green",
                frozenset(["Yellow", "Dark Green"]): "Yellow-Green",
                frozenset(["Yellow", "Light Green"]): "Yellow-Green",
                frozenset(["Yellow", "Olive Green"]): "Yellow-Green",
                frozenset(["Yellow", "Metallic Green"]): "Yellow-Green"
            }
            
            combo = frozenset([primary, secondary])
            if combo in two_tone_pairs:
                return two_tone_pairs[combo]
            else:
                # If it's a different two-tone combination not in the specific palette,
                # fallback to the primary (most dominant) color.
                return primary
        elif len(detected_color_names) == 1:
            return detected_color_names[0]
            
        return "Unknown"

    def classify_brand_heuristic(self, crop, class_name):
        """
        Classifies vehicle brand using aspect-ratio and color features.
        """
        if crop is None or crop.size == 0:
            return "Unknown"

        h, w, _ = crop.shape
        aspect_ratio = w / float(h) if h > 0 else 1.0
        
        # Heuristic rules based on shape features and BBox attributes
        if class_name == "car":
            if aspect_ratio > 1.4:
                return "Toyota"
            elif aspect_ratio < 1.1:
                return "BMW"
            else:
                return "Honda"
        elif class_name == "truck":
            return "Isuzu"
        elif class_name == "bus":
            return "Scania"
        else:
            return "Unknown"

    def classify_brand(self, crop, class_name):
        """
        Classifies vehicle brand using TinyViT (PyTorch, ONNX, or TensorRT) or heuristic fallback.
        """
        if crop is None or crop.size == 0:
            return "Unknown"

        if class_name not in ["car", "truck", "bus", "motorcycle"]:
            return self.classify_brand_heuristic(crop, class_name)

        if not self.brand_enabled or self.brand_backend is None:
            return self.classify_brand_heuristic(crop, class_name)

        try:
            # Preprocess crop
            input_tensor = self._preprocess_image(crop)

            # Inference
            if self.brand_backend == 'pytorch':
                import torch
                tensor = torch.from_numpy(input_tensor).to(self.device)
                with torch.no_grad():
                    logits = self.pytorch_model(tensor)
                    probs = torch.softmax(logits, dim=-1)
                    conf, idx = probs.max(1)
                    conf = conf.item()
                    idx = idx.item()
                
            elif self.brand_backend == 'onnx':
                onnx_inputs = {self.onnx_session.get_inputs()[0].name: input_tensor}
                logits = self.onnx_session.run(None, onnx_inputs)[0]
                # Softmax on logits
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                idx = np.argmax(probs[0])
                conf = probs[0][idx]

            elif self.brand_backend == 'tensorrt':
                import pycuda.driver as cuda
                # Copy host to device
                np.copyto(self.trt_inputs[0]['host_mem'], input_tensor.ravel())
                cuda.memcpy_htod(self.trt_inputs[0]['dev_mem'], self.trt_inputs[0]['host_mem'])
                
                # Execute context
                self.trt_context.execute_v2(self.trt_allocations)
                
                # Copy device to host
                cuda.memcpy_dtoh(self.trt_outputs[0]['host_mem'], self.trt_outputs[0]['dev_mem'])
                logits = self.trt_outputs[0]['host_mem']
                
                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                idx = np.argmax(probs)
                conf = probs[idx]

            else:
                return self.classify_brand_heuristic(crop, class_name)

            # Map index to class name
            if conf >= self.brand_conf_threshold and idx < len(self.brand_classes):
                brand = self.brand_classes[idx]
                if brand.lower() == "unknown":
                    return self.classify_brand_heuristic(crop, class_name)
                return brand
            else:
                return self.classify_brand_heuristic(crop, class_name)

        except Exception as e:
            logger.error(f"Inference error in brand classification: {e}")
            return self.classify_brand_heuristic(crop, class_name)

    def classify(self, crop, class_name):
        """
        Main entry point for classifying color and brand of a cropped vehicle image.
        Returns: (brand, color, confidence)
        """
        if crop is None or crop.size == 0:
            return "Unknown", "Unknown", 0.0
            
        # Classify color (using K-Means and CIELab)
        color = self.classify_color(crop)
        
        # Classify brand
        brand = self.classify_brand(crop, class_name)
        
        return brand, color, 1.0
