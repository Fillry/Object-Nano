import logging
from src.utils import calculate_centroid, denormalize_point

logger = logging.getLogger("ITS_Direction")

def intersect(p1, p2, p3, p4):
    """
    Checks if line segment p1-p2 intersects with line segment p3-p4.
    p1, p2, p3, p4 are points [x, y].
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def is_point_in_bbox(pt, bbox):
    """Checks if a point [x, y] is inside a bounding box [x1, y1, x2, y2]."""
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def line_intersects_bbox(p1, p2, bbox):
    """
    Checks if a line segment p1-p2 intersects or touches a bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Check if either endpoint of the line is inside the bbox
    if is_point_in_bbox(p1, bbox) or is_point_in_bbox(p2, bbox):
        return True
        
    # 2. Check intersection with 4 boundary segments of the bbox
    top_edge = [[x1, y1], [x2, y1]]
    bottom_edge = [[x1, y2], [x2, y2]]
    left_edge = [[x1, y1], [x1, y2]]
    right_edge = [[x2, y1], [x2, y2]]
    
    if intersect(p1, p2, top_edge[0], top_edge[1]):
        return True
    if intersect(p1, p2, bottom_edge[0], bottom_edge[1]):
        return True
    if intersect(p1, p2, left_edge[0], left_edge[1]):
        return True
    if intersect(p1, p2, right_edge[0], right_edge[1]):
        return True
        
    return False

class DirectionDetector:
    def __init__(self, line_a_norm, line_b_norm):
        """
        line_a_norm: [[x1, y1], [x2, y2]] in normalized format
        line_b_norm: [[x1, y1], [x2, y2]] in normalized format
        """
        self.line_a_norm = line_a_norm
        self.line_b_norm = line_b_norm
        self.crossed_a = {}  # track_id -> timestamp or frame_id
        self.crossed_b = {}  # track_id -> timestamp or frame_id
        
        # Keep track of logged touches to print log only ONCE per track per line
        self.logged_touches_a = set()
        self.logged_touches_b = set()

    def update(self, track, frame_width, frame_height, camera_id="Unknown"):
        """
        Checks if BBox touches or crosses lines.
        Updates track.direction.
        Returns direction ('entry', 'exit', 'pass') if a crossing event just completed.
        """
        if len(track.history) < 1:
            return None

        # Denormalize lines for this frame size
        p_a1 = denormalize_point(self.line_a_norm[0], frame_width, frame_height)
        p_a2 = denormalize_point(self.line_a_norm[1], frame_width, frame_height)
        p_b1 = denormalize_point(self.line_b_norm[0], frame_width, frame_height)
        p_b2 = denormalize_point(self.line_b_norm[1], frame_width, frame_height)

        curr_bbox = track.bbox
        track_id = track.track_id
        
        # 1. Check if BBox touches Line A
        touches_a = line_intersects_bbox(p_a1, p_a2, curr_bbox)
        if touches_a:
            if track_id not in self.crossed_a:
                self.crossed_a[track_id] = len(track.history)
            if track_id not in self.logged_touches_a:
                self.logged_touches_a.add(track_id)
                # Changed to debug level to keep main log output clean
                logger.debug(f"[{camera_id}] ALERT: Track {track_id} ({track.class_name}) touched Line A")

        # 2. Check if BBox touches Line B
        touches_b = line_intersects_bbox(p_b1, p_b2, curr_bbox)
        if touches_b:
            if track_id not in self.crossed_b:
                self.crossed_b[track_id] = len(track.history)
            if track_id not in self.logged_touches_b:
                self.logged_touches_b.add(track_id)
                # Changed to debug level to keep main log output clean
                logger.debug(f"[{camera_id}] ALERT: Track {track_id} ({track.class_name}) touched Line B")

        # 3. Evaluate direction once we have some crossings
        if track_id in self.crossed_a and track_id in self.crossed_b:
            if track.direction == "pass": # Only update if not already set to entry/exit
                # If crossed A before B
                if self.crossed_a[track_id] < self.crossed_b[track_id]:
                    track.direction = "exit"
                    logger.info(f"[{camera_id}] Track {track_id} direction set to EXIT (Line A -> Line B)")
                    return "exit"
                else:
                    track.direction = "entry"
                    logger.info(f"[{camera_id}] Track {track_id} direction set to ENTRY (Line B -> Line A)")
                    return "entry"
                    
        # Fallback check for Y movement vector if they crossed at least one line but missed the other in time
        elif len(track.history) >= 5 and track.direction == "pass":
            first_pt = calculate_centroid(track.history[0])
            last_pt = calculate_centroid(track.history[-1])
            dy = last_pt[1] - first_pt[1]
            
            if track_id in self.crossed_a or track_id in self.crossed_b:
                if dy > 30: # Moving downwards
                    track.direction = "entry"
                    logger.info(f"[{camera_id}] Track {track_id} direction set to ENTRY (Fallback Vector)")
                    return "entry"
                elif dy < -30: # Moving upwards
                    track.direction = "exit"
                    logger.info(f"[{camera_id}] Track {track_id} direction set to EXIT (Fallback Vector)")
                    return "exit"

        return None

    def clean_track(self, track_id):
        """Cleans up internal state for a removed track."""
        if track_id in self.crossed_a:
            del self.crossed_a[track_id]
        if track_id in self.crossed_b:
            del self.crossed_b[track_id]
        if track_id in self.logged_touches_a:
            self.logged_touches_a.remove(track_id)
        if track_id in self.logged_touches_b:
            self.logged_touches_b.remove(track_id)
