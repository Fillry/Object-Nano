import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class ThreadedStreamReader:
    def __init__(self, camera_id, source, target_fps=5, use_jetson_hw_dec=False, latency=100):
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.use_jetson_hw_dec = use_jetson_hw_dec
        self.latency = latency
        
        # Threads status
        self.running = False
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.has_new_frame = False
        self.width = 1280  # Default fallback
        self.height = 720  # Default fallback
        self.fps_actual = 0.0
        
        # Check if source is local file or RTSP
        self.is_file = not (str(source).startswith("rtsp://") or str(source).startswith("http://"))
        
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Starts the capture thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Stream reader for {self.camera_id} started (Source: {self.source}, Target FPS: {self.target_fps})")

    def stop(self):
        """Stops the capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
        with self.lock:
            if self.cap:
                self.cap.release()
                
        logger.info(f"Stream reader for {self.camera_id} stopped.")

    def _connect(self):
        """Attempts to connect to the video source."""
        with self.lock:
            if self.cap:
                self.cap.release()
            
            # Check if source is RTSP and GStreamer acceleration is requested
            is_rtsp = str(self.source).startswith("rtsp://") or str(self.source).startswith("rtsp:")
            if self.use_jetson_hw_dec and is_rtsp:
                # NVDEC Hardware Accelerated Decoding Pipeline
                # Using GStreamer with Gst-NV Video Converters
                pipeline = (
                    f"rtspsrc location={self.source} latency={self.latency} ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
                )
                logger.info(f"Connecting to source using Jetson Hardware Accelerated (GStreamer) pipeline: {pipeline}")
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                logger.info(f"Connecting to source: {self.source}")
                self.cap = cv2.VideoCapture(self.source)
            
            if self.cap.isOpened():
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # For video files, keep the actual frame rate if target_fps isn't set, 
                # but we will enforce frame-skipping based on target_fps in the loop
                logger.info(f"Connected to {self.camera_id}. Resolution: {self.width}x{self.height}")
                return True
            else:
                logger.error(f"Failed to open source: {self.source}")
                return False

    def _capture_loop(self):
        """Continuous loop to read frames from stream."""
        connected = self._connect()
        retry_delay = 1.0
        
        last_frame_time = time.time()
        fps_calc_start = time.time()
        fps_frames = 0
        
        while self.running:
            if not connected:
                logger.warning(f"Connection lost for {self.camera_id}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                connected = self._connect()
                # Exponential backoff for reconnection
                retry_delay = min(retry_delay * 2, 30.0)
                continue
            
            retry_delay = 1.0 # Reset delay on successful connection
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                # If it's a file, we reached the end
                if self.is_file:
                    logger.info(f"Finished reading video file: {self.source}")
                    # Loop video if running offline test, or stop
                    # Let's loop for continuous test simulation
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(f"Failed to read frame from RTSP stream {self.camera_id}")
                    connected = False
                    continue
            
            self.frame_id += 1
            fps_frames += 1
            
            # Calculate actual FPS of reading
            now = time.time()
            if now - fps_calc_start >= 1.0:
                self.fps_actual = fps_frames / (now - fps_calc_start)
                fps_frames = 0
                fps_calc_start = now
                
            # If it's a local file, we need to respect target_fps by sleeping 
            # to simulate a live camera stream speed, otherwise it reads as fast as possible
            if self.is_file:
                elapsed = now - last_frame_time
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
                with self.lock:
                    self.frame = frame
                    self.has_new_frame = True
            else:
                # For RTSP, we skip frames to match target FPS
                # RTSP reads at full speed (e.g. 25 FPS) but we only keep frames at self.frame_interval
                elapsed = now - last_frame_time
                if elapsed >= self.frame_interval:
                    with self.lock:
                        self.frame = frame
                        self.has_new_frame = True
                    last_frame_time = now
                else:
                    # Drop frame (do nothing) to keep CPU low
                    pass

    def get_latest_frame(self):
        """Retrieves the latest frame. Returns (has_new_frame, frame, frame_id)."""
        with self.lock:
            has_new = self.has_new_frame
            self.has_new_frame = False # Reset flag
            return has_new, self.frame.copy() if self.frame is not None else None, self.frame_id
