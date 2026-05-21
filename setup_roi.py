import cv2
import argparse
import sys

# Global variables to store clicked points
clicked_points = []
img_display = None
img_original = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_display, img_original
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w, _ = img_original.shape
        # Calculate normalized coordinates
        norm_x = round(x / w, 4)
        norm_y = round(y / h, 4)
        
        # Save points
        clicked_points.append(((x, y), (norm_x, norm_y)))
        print(f"Point {len(clicked_points)}: Pixel=[{x}, {y}] | Normalized=[{norm_x}, {norm_y}]")
        
        # Redraw
        redraw_image()

def redraw_image():
    global clicked_points, img_display, img_original
    img_display = img_original.copy()
    
    # Draw points and lines
    for i, ((px, py), (nx, ny)) in enumerate(clicked_points):
        # Draw dot
        cv2.circle(img_display, (px, py), 5, (0, 0, 255), -1)
        # Draw text label
        cv2.putText(img_display, str(i+1), (px + 8, py - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw lines connecting the points for polygon visualization
        if i > 0:
            cv2.line(img_display, clicked_points[i-1][0], (px, py), (0, 255, 0), 2)
            
    # For polygons, connect the last point back to the first one if there are > 2 points
    if len(clicked_points) > 2:
        cv2.line(img_display, clicked_points[-1][0], clicked_points[0][0], (255, 0, 0), 1)
        
    cv2.imshow("ROI Setup Tool", img_display)

def main():
    global img_display, img_original, clicked_points
    
    parser = argparse.ArgumentParser(description="Interactive ROI and Line Point Extractor")
    parser.add_argument("--source", type=str, default="vdo1_8-39-48_.mp4", 
                        help="Video source or RTSP stream URL")
    args = parser.parse_args()

    print(f"Opening video source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        sys.exit(1)
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Error: Could not read frame from source")
        sys.exit(1)
        
    img_original = frame
    img_display = img_original.copy()
    h, w, c = img_original.shape
    
    print("\n" + "="*60)
    print(" INSTRUCTIONS FOR ROI SETUP TOOL:")
    print(" - Click Left Mouse Button: Add a point")
    print(" - Press 'c': Clear all points")
    print(" - Press 'p': Print Polygon ROI Coordinates (for roi_mask)")
    print(" - Press 'l': Print Line Coordinates (uses last 2 clicked points)")
    print(" - Press 'q' or 'ESC': Quit")
    print("="*60 + "\n")
    
    cv2.namedWindow("ROI Setup Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Setup Tool", 1280, 720) # Resize window for convenient display
    cv2.imshow("ROI Setup Tool", img_display)
    cv2.setMouseCallback("ROI Setup Tool", mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # ESC
            break
        elif key == ord('c'):
            clicked_points = []
            img_display = img_original.copy()
            cv2.imshow("ROI Setup Tool", img_display)
            print("Cleared all points.")
        elif key == ord('p'):
            if len(clicked_points) < 3:
                print("Warning: You need at least 3 points to define a Polygon ROI mask.")
                continue
                
            print("\n" + "-"*40)
            print("COPY THIS FOR 'roi_mask' in camera_config.yaml:")
            
            # Print Normalized
            norm_str = ", ".join([f"[{pt[1][0]}, {pt[1][1]}]" for pt in clicked_points])
            print(f"roi_mask: [{norm_str}]")
            
            # Print Pixels (for manual reference)
            pixel_str = ", ".join([f"[{pt[0][0]}, {pt[0][1]}]" for pt in clicked_points])
            print(f"roi_mask (Pixel reference): [{pixel_str}]")
            print("-"*40 + "\n")
            
        elif key == ord('l'):
            if len(clicked_points) < 2:
                print("Warning: You need at least 2 points to define a line.")
                continue
                
            # Take the last two points clicked
            pt1 = clicked_points[-2]
            pt2 = clicked_points[-1]
            
            print("\n" + "-"*40)
            print("COPY THIS FOR 'line_a' or 'line_b' in camera_config.yaml:")
            print(f"line_coords: [[{pt1[1][0]}, {pt1[1][1]}], [{pt2[1][0]}, {pt2[1][1]}]]")
            print(f"line_coords (Pixel reference): [[{pt1[0][0]}, {pt1[0][1]}], [{pt2[0][0]}, {pt2[0][1]}]]")
            print("-"*40 + "\n")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
