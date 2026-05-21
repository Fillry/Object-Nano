import numpy as np
import math

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

def calculate_centroid_dist(box1, box2):
    """Calculates Euclidean distance between centroids of two boxes."""
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

class TrackedObject:
    def __init__(self, track_id, bbox, class_name, confidence):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        
        self.age = 0  # Number of consecutive frames tracked
        self.time_since_update = 0  # Number of frames since last update
        self.history = [bbox]  # History of bounding boxes
        
        # Metadata fields for Classification result caching
        self.color = "Unknown"
        self.brand = "Unknown"
        self.direction = "pass"
        self.is_logged = False # Whether this track is already logged in DB
        
        # Best crop tracking for Classification input
        self.best_crop = None
        self.best_crop_score = 0.0

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.age += 1
        self.history.append(bbox)
        if len(self.history) > 150: # Increased history slightly
            self.history.pop(0)

class IoUTracker:
    def __init__(self, max_age=60, min_hits=2, iou_threshold=0.2, max_centroid_dist=200):
        """
        max_age: Number of frames to keep a lost track active (increased for robust tracking)
        min_hits: Number of frames before a track is active (decreased to 2 for faster acquisition)
        iou_threshold: Minimum IoU to associate (decreased to 0.2 to handle fast movements)
        max_centroid_dist: Max pixel distance to match using centroids if IoU is zero (useful for fast motorcycles)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        
        self.track_id_counter = 0
        self.tracks = [] # List of TrackedObject

    def update(self, detections):
        """
        Updates the tracker with new detections using IoU followed by Centroid distance fallback.
        """
        # Increment time since update for all existing tracks
        for track in self.tracks:
            track.time_since_update += 1
            
        matched_detections = set()
        matched_tracks = set()
        
        # --- Stage 1: Match detections based on IoU ---
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    if track.class_name == det['class_name']:
                        iou_matrix[t_idx, d_idx] = calculate_iou(track.bbox, det['bbox'])
                    else:
                        iou_matrix[t_idx, d_idx] = 0.0
            
            # Greedy matching for IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                det = detections[d_idx]
                self.tracks[t_idx].update(det['bbox'], det['conf'])
                
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # --- Stage 2: Fallback matching using Centroid Distance for remaining ---
        # Crucial for small, fast objects like motorcycles where IoU is 0 but centroids are close
        unmatched_tracks = [t_idx for t_idx in range(len(self.tracks)) if t_idx not in matched_tracks]
        unmatched_detections = [d_idx for d_idx in range(len(detections)) if d_idx not in matched_detections]
        
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            dist_matrix = np.full((len(unmatched_tracks), len(unmatched_detections)), float('inf'), dtype=np.float32)
            
            for i, t_idx in enumerate(unmatched_tracks):
                track = self.tracks[t_idx]
                for j, d_idx in enumerate(unmatched_detections):
                    det = detections[d_idx]
                    if track.class_name == det['class_name']:
                        dist_matrix[i, j] = calculate_centroid_dist(track.bbox, det['bbox'])
            
            # Greedy matching for Centroid Distance
            while True:
                min_val = np.min(dist_matrix)
                if min_val > self.max_centroid_dist:
                    break
                
                i_idx, j_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                
                t_idx = unmatched_tracks[i_idx]
                d_idx = unmatched_detections[j_idx]
                
                det = detections[d_idx]
                self.tracks[t_idx].update(det['bbox'], det['conf'])
                
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                
                dist_matrix[i_idx, :] = float('inf')
                dist_matrix[:, j_idx] = float('inf')

        # --- Stage 3: Create new tracks for unmatched detections ---
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                self.track_id_counter += 1
                new_track = TrackedObject(
                    track_id=self.track_id_counter,
                    bbox=det['bbox'],
                    class_name=det['class_name'],
                    confidence=det['conf']
                )
                self.tracks.append(new_track)

        # Filter out old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        
        # Return active tracks
        active_tracks = [
            track for track in self.tracks 
            if track.time_since_update == 0 and (track.age >= self.min_hits or track.time_since_update == 0)
        ]
        
        return active_tracks
