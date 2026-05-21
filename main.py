import os
import cv2
import yaml
import time
import argparse
import logging
from datetime import datetime

from src.stream_reader import ThreadedStreamReader
from src.detector import YOLODetector
from src.tracker import IoUTracker
from src.classifier import VehicleClassifier
from src.direction import DirectionDetector
from src.database import DatabaseLogger
from src.utils import ensure_dir, draw_info, draw_bbox_with_label, denormalize_polygon, is_point_inside_polygon, calculate_centroid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ITS_Main")

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Edge AI for Intelligent Transport System (ITS)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--db", type=str, default="its_database.db", help="Path to SQLite database")
    args = parser.parse_args()

    # Load configurations
    camera_config = load_yaml("configs/camera_config.yaml")
    model_config = load_yaml("configs/model_config.yaml")
    
    # Initialize DB Logger
    db_logger = DatabaseLogger(db_path=args.db)
    db_logger.start()

    # Initialize YOLO Detector
    detector = YOLODetector(
        model_path=model_config['detector']['model_path'],
        conf_threshold=model_config['detector']['conf_threshold'],
        iou_threshold=model_config['detector']['iou_threshold'],
        device=model_config['detector']['device']
    )

    # Initialize Classifiers
    classifier = VehicleClassifier(model_config['classifiers'])

    # Initialize Camera Pipelines
    streams = {}
    trackers = {}
    direction_detectors = {}
    vehicle_counts = {} # camera_id -> {type: count}
    for cam_id, cam_info in camera_config['cameras'].items():
        # Get stream reader configurations safely
        sr_config = model_config.get('stream_reader', {})
        use_hw = sr_config.get('use_jetson_hw_dec', False)
        latency = sr_config.get('latency', 100)
        
        # Setup Reader
        reader = ThreadedStreamReader(
            camera_id=cam_id,
            source=cam_info['source'],
            target_fps=cam_info['fps'],
            use_jetson_hw_dec=use_hw,
            latency=latency
        )
        reader.start()
        streams[cam_id] = reader
        
        # Setup Tracker
        trackers[cam_id] = IoUTracker(
            max_age=model_config['tracker']['max_age'],
            min_hits=model_config['tracker']['min_hits'],
            iou_threshold=model_config['tracker']['iou_threshold']
        )
        
        # Setup Direction Detector
        direction_detectors[cam_id] = DirectionDetector(
            line_a_norm=cam_info['line_a'],
            line_b_norm=cam_info['line_b']
        )
        
        vehicle_counts[cam_id] = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    # Tracking list to save the first 5 unique detected IDs' crops
    saved_ids = set()
    output_crop_dir = "cropped_samples"

    logger.info("ITS Pipeline initialized. Processing streams...")
    
    try:
        while True:
            loop_start = time.time()
            active_cams = 0
            
            for cam_id, reader in streams.items():
                has_new, frame, frame_id = reader.get_latest_frame()
                if not has_new or frame is None:
                    continue
                
                active_cams += 1
                h, w, _ = frame.shape
                
                # 1. Run Object Detection
                detections = detector.detect(frame)
                
                # 2. Filter Detections using ROI Mask if specified in config
                cam_info = camera_config['cameras'][cam_id]
                filtered_detections = []
                
                if 'roi_mask' in cam_info:
                    roi_poly = denormalize_polygon(cam_info['roi_mask'], w, h)
                    for det in detections:
                        # Check if centroid is inside ROI
                        cx, cy = calculate_centroid(det['bbox'])
                        if is_point_inside_polygon((cx, cy), roi_poly):
                            filtered_detections.append(det)
                else:
                    filtered_detections = detections
                
                # 3. Update Tracker
                active_tracks = trackers[cam_id].update(filtered_detections)
                
                # 4. Process each active track
                for track in active_tracks:
                    # Update direction estimation
                    new_direction = direction_detectors[cam_id].update(track, w, h, camera_id=cam_id)
                    
                    # Update best crop for classification (highest confidence or largest bbox)
                    x1, y1, x2, y2 = track.bbox
                    # Clamp coordinates to frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop_w, crop_h = x2 - x1, y2 - y1
                    crop_area = crop_w * crop_h
                    
                    # If this is the largest or highest-conf crop so far, update the best crop
                    if crop_area > 400 and (track.best_crop is None or track.confidence > track.best_crop_score):
                        track.best_crop = frame[y1:y2, x1:x2]
                        track.best_crop_score = track.confidence
                    
                    # LOGGING CRITERIA:
                    # Scenario A: The track has completed a Line Crossing (new_direction detected)
                    # Scenario B: The track is stable (tracked for at least 15 frames) but not logged yet
                    should_log = False
                    
                    if not track.is_logged:
                        if track.direction != "pass":
                            # We just crossed a line
                            should_log = True
                        elif track.age >= 15:
                            # Fallback: stable track but hasn't crossed line (or skipped line)
                            should_log = True
                            
                    if should_log:
                        # Classify Brand and Color on the best crop
                        brand, color, _ = classifier.classify(track.best_crop, track.class_name)
                        track.brand = brand
                        track.color = color
                        
                        # Increment stats count
                        v_type = track.class_name
                        if v_type in vehicle_counts[cam_id]:
                            vehicle_counts[cam_id][v_type] += 1
                            
                        # Timestamp of log
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Log to database (INSERT)
                        db_logger.log_vehicle(
                            camera_id=cam_id,
                            track_id=track.track_id,
                            timestamp=now_str,
                            vehicle_type=track.class_name,
                            brand=track.brand,
                            color=track.color,
                            direction=track.direction,
                            confidence=track.confidence,
                            bbox=track.bbox
                        )
                        track.is_logged = True
                        logger.info(f"[{cam_id}] Logged ID {track.track_id}: {track.color} {track.brand} {track.class_name} going {track.direction} (BBox: {track.bbox})")
                        
                        # --- FEATURE: Save the first 5 unique detected vehicle IDs' Full and Sub Crops ---
                        # Create unique identifier based on camera_id and track_id to avoid collision across camera streams
                        unique_id_str = f"{cam_id}_ID{track.track_id}"
                        if len(saved_ids) < 5 and unique_id_str not in saved_ids:
                            saved_ids.add(unique_id_str)
                            os.makedirs(output_crop_dir, exist_ok=True)
                            
                            # Prepare Full Crop
                            full_crop = track.best_crop
                            fh, fw, _ = full_crop.shape
                            
                            # Prepare Sub Crop matching classifier.py crop ratio
                            sy1, sy2 = int(fh * 0.35), int(fh * 0.75)
                            sx1, sx2 = int(fw * 0.20), int(fw * 0.80)
                            sub_crop = full_crop[sy1:sy2, sx1:sx2]
                            
                            color_name = track.color.replace(" ", "_")
                            # Save files with requested name format
                            full_path = os.path.join(output_crop_dir, f"ID{track.track_id}_Full_Crop_{color_name}.jpg")
                            sub_path = os.path.join(output_crop_dir, f"ID{track.track_id}_Sub_Crop_{color_name}.jpg")
                            
                            if full_crop is not None and full_crop.size > 0:
                                cv2.imwrite(full_path, full_crop)
                            if sub_crop is not None and sub_crop.size > 0:
                                cv2.imwrite(sub_path, sub_crop)
                                
                            logger.info(f"Saved crop samples to: {full_path} and {sub_path}")
                    
                    # Scenario C: The track was ALREADY logged (e.g. as 'pass') but now we got a real direction update ('entry' or 'exit')
                    elif track.is_logged and new_direction is not None and new_direction != "pass":
                        db_logger.update_vehicle_direction(
                            camera_id=cam_id,
                            track_id=track.track_id,
                            direction=new_direction
                        )
                        # We do NOT log a new row, we just update the direction in the DB and logger
                        logger.info(f"[{cam_id}] UPDATED direction for Logged ID {track.track_id} to: {new_direction} (BBox: {track.bbox})")
                
                # 5. Visualization (Skip if headless)
                if not args.headless:
                    # Draw ROI
                    if 'roi_mask' in cam_info:
                        roi_poly = denormalize_polygon(cam_info['roi_mask'], w, h)
                        cv2.polylines(frame, [roi_poly], True, (255, 255, 0), 1)
                        
                    # Draw Lines
                    p_a1 = denormalize_polygon([cam_info['line_a'][0]], w, h)[0]
                    p_a2 = denormalize_polygon([cam_info['line_a'][1]], w, h)[0]
                    p_b1 = denormalize_polygon([cam_info['line_b'][0]], w, h)[0]
                    p_b2 = denormalize_polygon([cam_info['line_b'][1]], w, h)[0]
                    cv2.line(frame, tuple(p_a1), tuple(p_a2), (0, 0, 255), 2) # Line A Red
                    cv2.line(frame, tuple(p_b1), tuple(p_b2), (255, 0, 0), 2) # Line B Blue
                    
                    # Draw active tracks
                    for track in active_tracks:
                        label = f"ID:{track.track_id} {track.class_name}"
                        if track.is_logged:
                            label += f" ({track.color} {track.brand})"
                        
                        # Change color based on direction
                        draw_color = (0, 255, 0) # Green default
                        if track.direction == "entry":
                            draw_color = (255, 0, 0) # Blue
                        elif track.direction == "exit":
                            draw_color = (0, 0, 255) # Red
                            
                        frame = draw_bbox_with_label(frame, track.bbox, label, draw_color)
                        
                    # Draw Stats overlay
                    frame = draw_info(frame, cam_id, reader.fps_actual, len(active_tracks), vehicle_counts[cam_id])
                    
                    # Show Frame
                    cv2.imshow(f"ITS Monitor - {cam_id}", frame)
                    
            if not args.headless:
                # Key listener to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Idle sleeping to regulate loop speed if no new frames in any cameras
            if active_cams == 0:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    finally:
        # Stop all threads and release resources
        for reader in streams.values():
            reader.stop()
            
        db_logger.stop()
        cv2.destroyAllWindows()
        logger.info("System shut down cleanly.")

if __name__ == "__main__":
    main()
