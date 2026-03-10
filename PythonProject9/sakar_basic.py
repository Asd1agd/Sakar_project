import cv2
import time
from ultralytics import YOLO
import closest_cluster as cl
import math

# ------------------------------------------------------------
# Helper: draw text with a solid background for readability
# ------------------------------------------------------------
def draw_text_with_background(img, text, pos, font_scale=0.5,
                              text_color=(255,255,255), bg_color=(0,0,0),
                              thickness=2, padding=3):
    """Draw text with a solid background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Background rectangle
    cv2.rectangle(img,
                  (x, y - text_h - padding),
                  (x + text_w + padding*2, y + baseline + padding),
                  bg_color, -1)
    # Text
    cv2.putText(img, text, (x + padding, y),
                font, font_scale, text_color, thickness)

# ------------------------------------------------------------
# Load your custom trained model
# ------------------------------------------------------------
model = YOLO('best (3).pt')

# Path to input video
video_path = 'WhatsApp Video 2026-03-08 at 12.53.01 PM.mp4'
# video_path = 'WhatsApp Video 2026-03-08 at 5.43.19 PM.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for output writer
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter to save output
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (width, height))

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0
frame_count = 0
fps_display = 0

closest = None
closest_id = None
virtual_box_passed_count = 0
object_passed_ids = {}

track_status_dict = {}
track_pose_depth_dict = {}

# Example usage with 2D points
clusterer = cl.GridClusterCOM(com_radius=70, reduction_threshold=4, frame_threshold=80)

def get_depth(arr,frame):
    x1, y1, x2, y2 = arr[0], arr[1], arr[2], arr[3]
    cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
    area = calculate_area(frame, x1, y1, x2, y2)
    depth = get_depth_from_area(area)
    return depth

def calculate_area(frame, x1, y1, x2, y2):
    frame_copy = frame.copy()
    area = abs(x2 - x1) * abs(y2 - y1)
    return area

def get_depth_from_area(area):
    caliberation_area_for_1m = 10000 * 1.5  # example area corresponding to 1 meter distance
    distance = 1.0 * (caliberation_area_for_1m / area) ** 0.5  # simple inverse square relationship
    return distance

def inside_virtual_box(track_pose_depth, virtual_box):
    x1, y1, x2, y2, depth = track_pose_depth
    cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
    if cx > virtual_box[0] and cx < virtual_box[2] and cy > virtual_box[1] and cy < virtual_box[3]:
        return True
    else:
        return False

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    print("frame_count:", frame_count)

    # Time when frame is captured (start of processing)
    new_frame_time = time.time()

    # ------------------------------------------------------------------
    # Create a masked version of the frame: black out the top 15%
    # ------------------------------------------------------------------
    h, w = frame.shape[:2]
    top_region_height = int(0.15 * h)  # 15% of frame height
    masked_frame = frame.copy()
    masked_frame[:top_region_height, :] = 0  # set top region to black

    # Run inference on the masked frame
    results = model(masked_frame)

    # Make a copy of the original frame to draw on
    annotated_frame = frame.copy()

    # ------------------------------------------------------------
    # Manual drawing of bounding boxes with class name and confidence
    # ------------------------------------------------------------
    if results[0].boxes is not None:                 # Check if any detections exist
        closest = None
        closest_id = None
        virtual_box_inside_count = 0
        print("closest:", closest, " ", closest_id)
        for box in results[0].boxes:
            # Get bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence and class id
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]          # Retrieve class name from model

            cx, cy = (x2 + x1) / 2, (y2 + y1) / 2

            depth = get_depth([x1, y1, x2, y2], frame)

            track_point, track_id = clusterer.add_point([cx, cy], frame_count)

            track_status_dict[track_id] = track_status_dict.get(track_id, 0) + 1
            track_status_num = track_status_dict[track_id]
            track_status = "Not traked"
            if track_status_num > max(4 * int(fps_display), 10):
                track_status = "Active"

            track_pose_depth_dict[track_id] = [x1, y1, x2, y2, depth]
            if closest is None:
                closest = depth
                closest_id = track_id
            if depth < closest:
                closest = depth
                closest_id = track_id
            print("closest:", closest, " ", closest_id)

            # --------------------------------------------------------
            # Virtual box definition (fixed position)
            # --------------------------------------------------------
            x1_virtual, y1_virtual, x2_virtual, y2_virtual = [200, 200, 500, 500]
            virtual_box = [x1_virtual, y1_virtual, x2_virtual, y2_virtual]
            cv2.rectangle(annotated_frame,
                          (x1_virtual, y1_virtual),
                          (x2_virtual, y2_virtual),
                          (255, 255, 0), 3)   # cyan for virtual box

            # Check if object inside virtual box
            if inside_virtual_box(track_pose_depth_dict[track_id], virtual_box):
                virtual_box_inside_count += 1
                if track_id not in object_passed_ids:
                    object_passed_ids[track_id] = 0
                    virtual_box_passed_count += 1

            # --------------------------------------------------------
            # Determine bounding box color based on status
            # --------------------------------------------------------
            if track_id == closest_id:
                box_color = (0, 0, 255)        # red for closest
                box_thickness = 3
            elif track_status == "Active":
                box_color = (0, 255, 0)        # green for stable track
                box_thickness = 2
            else:
                box_color = (0, 165, 255)      # orange for new/unstable
                box_thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw center point (small white outline + red fill)
            cv2.circle(annotated_frame, (int(track_point[0]), int(track_point[1])), 6, (255,255,255), 2)
            cv2.circle(annotated_frame, (int(track_point[0]), int(track_point[1])), 4, (0,0,255), -1)

            # --------------------------------------------------------
            # Prepare and draw text labels with background
            # --------------------------------------------------------
            label1 = f"{class_name} {conf:.2f} ID:{track_id}"
            label2 = f"{track_status} depth:{depth:.2f}"

            # Place labels above the box (or inside if enough space)
            if y2 - y1 > 50:   # enough height to put first label inside
                pos1 = (x1 + 5, y1 + 20)
            else:
                pos1 = (x1, y1 - 10)
            pos2 = (x1, y1 - 30)   # second label above first

            draw_text_with_background(annotated_frame, label1, pos1,
                                      bg_color=box_color, text_color=(255,255,255))
            draw_text_with_background(annotated_frame, label2, pos2,
                                      bg_color=(50,50,50), text_color=(200,200,200))

            # --------------------------------------------------------
            # Virtual box info
            # --------------------------------------------------------
            info_text = f"inside:{virtual_box_inside_count} passed:{virtual_box_passed_count}"
            info_pos = (x1_virtual, y1_virtual - 10)
            draw_text_with_background(annotated_frame, info_text, info_pos,
                                      bg_color=(255,255,0), text_color=(0,0,0))

        # ------------------------------------------------------------
        # After loop over detections: highlight closest object again
        # ------------------------------------------------------------
        if closest_id is not None:
            x1, y1, x2, y2, depth = track_pose_depth_dict[closest_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Add "closest" label with background
            draw_text_with_background(annotated_frame, "closest",
                                      (x1, y1 + 20), bg_color=(0,0,255), text_color=(255,255,255))

    # ------------------------------------------------------------
    # FPS display
    # ------------------------------------------------------------
    if prev_frame_time != 0:
        fps_display = 1 / (new_frame_time - prev_frame_time)
        fps_display = int(fps_display)
    else:
        fps_display = 0

    draw_text_with_background(annotated_frame, f"FPS: {fps_display}", (10, 30),
                              font_scale=1, bg_color=(0,0,0), text_color=(0,255,0))

    # Update previous time for next frame
    prev_frame_time = new_frame_time

    # Display the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Write to output video
    out.write(annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()