import os
import cv2
import numpy as np

def ensure_dir(path):
    """Ensures that a directory exists."""
    os.makedirs(path, exist_ok=True)

def denormalize_point(norm_pt, width, height):
    """Converts a normalized point [0.0 - 1.0] to pixel coordinates [0 - width/height]."""
    return [int(norm_pt[0] * width), int(norm_pt[1] * height)]

def denormalize_polygon(norm_polygon, width, height):
    """Converts a normalized polygon array to pixel coordinates."""
    return np.array([denormalize_point(pt, width, height) for pt in norm_polygon], dtype=np.int32)

def calculate_centroid(bbox):
    """Calculates the center point (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def is_point_inside_polygon(pt, polygon):
    """Checks if a point (x, y) is inside a polygon using OpenCV."""
    # polygon must be a numpy array of shape (N, 1, 2) or (N, 2)
    result = cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False)
    return result >= 0

def draw_info(frame, camera_id, fps, active_tracks, vehicle_counts=None):
    """Draws basic statistics on the top-left of the frame."""
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    
    cv2.rectangle(frame, (10, 10), (280, 100), bg_color, -1)
    cv2.putText(frame, f"Cam: {camera_id}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Active Tracks: {active_tracks}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    if vehicle_counts:
        # Draw counts below the main stats box
        y_offset = 120
        cv2.rectangle(frame, (10, y_offset - 15), (280, y_offset + 100), bg_color, -1)
        for i, (k, v) in enumerate(vehicle_counts.items()):
            cv2.putText(frame, f"{k.capitalize()}: {v}", (20, y_offset + (i * 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
    return frame

def draw_bbox_with_label(frame, bbox, label, color=(0, 255, 0)):
    """Draws bounding box with a clean label background."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Text background
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w + 6, y1), color, -1)
    
    cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
