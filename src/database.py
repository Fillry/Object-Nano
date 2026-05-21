import sqlite3
import threading
import queue
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseLogger:
    def __init__(self, db_path="its_database.db"):
        self.db_path = db_path
        self.log_queue = queue.Queue()
        self.is_running = False
        self.writer_thread = None
        self._init_db()

    def _init_db(self):
        """Initializes the database schema and enables WAL mode."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable Write-Ahead Logging (WAL) for concurrency & safety
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            vehicle_type TEXT NOT NULL,
            brand TEXT DEFAULT 'Unknown',
            color TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER
        );
        """)
        
        # Create Index for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_time ON vehicle_logs(camera_id, timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_id ON vehicle_logs(track_id);")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at {self.db_path}")

    def start(self):
        """Starts the background DB writer thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self.writer_thread.start()
        logger.info("Database Logger thread started.")

    def stop(self):
        """Stops the background DB writer thread and flushes remaining queue items."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=3.0)
        logger.info("Database Logger thread stopped.")

    def log_vehicle(self, camera_id, track_id, timestamp, vehicle_type, brand, color, direction, confidence, bbox):
        """Queue a vehicle detection log for async writing to DB."""
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)
            
        x1, y1, x2, y2 = bbox
        
        log_entry = (
            camera_id,
            int(track_id),
            timestamp_str,
            vehicle_type,
            brand,
            color,
            direction,
            float(confidence),
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )
        # Put with command type "INSERT"
        self.log_queue.put(("INSERT", log_entry))

    def update_vehicle_direction(self, camera_id, track_id, direction):
        """Queue a direction update for an existing vehicle log."""
        self.log_queue.put(("UPDATE_DIR", (direction, camera_id, int(track_id))))

    def _db_writer_worker(self):
        """Worker thread that writes logs from queue to the SQLite DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        while self.is_running or not self.log_queue.empty():
            try:
                # Retrieve command from queue
                try:
                    cmd_type, data = self.log_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if cmd_type == "INSERT":
                    cursor.execute("""
                    INSERT INTO vehicle_logs (
                        camera_id, track_id, timestamp, vehicle_type, brand, color, direction, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, data)
                    
                elif cmd_type == "UPDATE_DIR":
                    # Update the direction of the latest entry matching camera_id and track_id
                    cursor.execute("""
                    UPDATE vehicle_logs 
                    SET direction = ? 
                    WHERE camera_id = ? AND track_id = ?
                    """, data)
                    
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error executing DB transaction: {e}")
                try:
                    conn.close()
                except:
                    pass
                time.sleep(2.0)
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
        conn.close()
        logger.info("Database connection closed gracefully.")
