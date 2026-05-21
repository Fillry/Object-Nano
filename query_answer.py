import argparse
import sqlite3
from datetime import datetime

def print_table(headers, rows):
    """Helper to format and print a table using plain text."""
    if not rows:
        print("No records found.")
        return
        
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(str(val)))
            
    # Format line separator
    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    
    # Print header
    print(sep)
    header_str = "|" + "|".join([f" {headers[idx]:<{widths[idx]}} " for idx in range(len(headers))]) + "|"
    print(header_str)
    print(sep)
    
    # Print rows
    for row in rows:
        row_str = "|" + "|".join([f" {str(row[idx]):<{widths[idx]}} " for idx in range(len(row))]) + "|"
        print(row_str)
        
    print(sep)

def query_its(db_path, camera_id, start_time, end_time, vehicle_type=None, direction=None):
    """
    Connects to the SQLite DB and executes query to find vehicles matching criteria.
    Displays BBox coordinates [x1, y1, x2, y2] in the results table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prepare query with BBox coordinates
    query = """
    SELECT 
        camera_id,
        track_id,
        timestamp,
        vehicle_type,
        brand,
        color,
        direction,
        bbox_x1,
        bbox_y1,
        bbox_x2,
        bbox_y2
    FROM vehicle_logs
    WHERE camera_id = ?
    """
    params = [camera_id]
    
    # Handle time/datetime filters
    if len(start_time) == 8 and ":" in start_time: # Format: "HH:MM:SS"
        query += " AND time(timestamp) BETWEEN time(?) AND time(?)"
        params.extend([start_time, end_time])
    else: # Format: "YYYY-MM-DD HH:MM:SS"
        query += " AND timestamp BETWEEN ? AND ?"
        params.extend([start_time, end_time])
        
    if vehicle_type:
        query += " AND vehicle_type = ?"
        params.append(vehicle_type.lower())
        
    if direction:
        query += " AND direction = ?"
        params.append(direction.lower())
        
    query += " ORDER BY timestamp ASC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    headers = ["Camera", "Track ID", "Timestamp", "Type", "Brand", "Color", "Direction", "BBox"]
    
    print("\n" + "="*85)
    print(f" ITS REPORT: {camera_id} | Range: {start_time} to {end_time}")
    if vehicle_type:
        print(f" Filter Type: {vehicle_type}")
    if direction:
        print(f" Filter Direction: {direction}")
    print("="*85)
    
    if not rows:
        print("No vehicle records matched your search criteria.")
        print("="*85 + "\n")
        conn.close()
        return
        
    # Format bbox coordinates into [x1, y1, x2, y2] format
    formatted_rows = []
    for r in rows:
        # Build bbox coordinate string
        bbox_str = f"[{r[7]}, {r[8]}, {r[9]}, {r[10]}]"
        # Create a new row entry replacing separate coords with bbox string
        formatted_rows.append(r[:7] + (bbox_str,))
        
    # Print results table
    print_table(headers, formatted_rows)
    
    # Summaries
    unique_ids = set(r[1] for r in rows)
    total_unique_vehicles = len(unique_ids)
    print(f"Total Unique Vehicles Counted: {total_unique_vehicles}")
    
    # Simple aggregation in pure Python
    brands = {}
    colors = {}
    types = {}
    
    # Use track_id to aggregate to avoid double-counting in segments
    seen_tracks = set()
    for r in rows:
        t_id, v_type, brand, color = r[1], r[3], r[4], r[5]
        if t_id not in seen_tracks:
            seen_tracks.add(t_id)
            brands[brand] = brands.get(brand, 0) + 1
            colors[color] = colors.get(color, 0) + 1
            types[v_type] = types.get(v_type, 0) + 1
            
    print("\nBrand Breakdown:")
    for b, c in sorted(brands.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {b}: {c} vehicle(s)")
        
    print("\nColor Breakdown:")
    for col, c in sorted(colors.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {col}: {c} vehicle(s)")
        
    print("\nVehicle Type Breakdown:")
    for t, c in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {t}: {c} vehicle(s)")

    print("="*85 + "\n")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Query Engine for ITS Evaluation")
    parser.add_argument("--db", type=str, default="its_database.db", help="Path to SQLite database")
    parser.add_argument("--camera", type=str, required=True, help="Camera ID (e.g. CCTV01, CCTV02)")
    parser.add_argument("--start", type=str, required=True, help="Start time (Format 'HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS')")
    parser.add_argument("--end", type=str, required=True, help="End time (Format 'HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS')")
    parser.add_argument("--type", type=str, default=None, help="Filter by vehicle type (car, motorcycle, bus, truck)")
    parser.add_argument("--direction", type=str, default=None, help="Filter by direction (entry, exit, pass)")
    
    args = parser.parse_args()
    
    query_its(
        db_path=args.db,
        camera_id=args.camera,
        start_time=args.start,
        end_time=args.end,
        vehicle_type=args.type,
        direction=args.direction
    )

if __name__ == "__main__":
    main()
