#Load Libraries
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import LineString, Point
from collections import defaultdict

#Initialize Model
model = YOLO("yolov8n.pt")

#Initialize class names
class_names = {3: "Motorcycle", 2: "car", 5: "bus", 7: "truck"}

# Initialize class_wise_count for all classes
class_wise_count = {name: {"total": 0} for name in class_names.values()}

# Create a VideoCapture object for RTSP stream
cap = cv2.VideoCapture("C:\\Users\\asus\\Downloads\\traffictrim.mp4")

# Define region points
reg_pts = [(80, 400), (1230, 400)]
track_history = defaultdict(list)
draw_tracks = True  # Ensure track drawing is enabled
track_color = None
count_ids = []
counting_region = LineString(reg_pts)
total_count = 0
annotator = None

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    cv2.rectangle(im0, (600, 740), (1480, 780), (255, 255, 255), -1)
    tracks = model.track(im0, persist=True, show=False, tracker="bytetrack.yaml", classes=[2, 3, 5, 7], conf=0.30)
    annotator = Annotator(im0, 2, class_names)
    annotator.draw_region(reg_pts=reg_pts, color=(0, 255, 0), thickness=2)

    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Draw bounding box
            annotator.box_label(box, label=f"{class_names[cls]}", color=colors(int(track_id), True))

            # Draw Tracks
            track_line = track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 10:
                track_line.pop(0)

            # Draw track trails
            if draw_tracks:
                annotator.draw_centroid_and_tracks(
                    track_line,
                    color=track_color if track_color else colors(int(track_id), True),
                    track_thickness=2,
                )

            #Counting vehicles
            prev_position = track_history[track_id][-2] if len(track_history[track_id]) > 1 else None
            if len(reg_pts) == 2:
                if prev_position is not None and track_id not in count_ids:
                    distance = Point(track_line[-1]).distance(counting_region)
                    if distance < 15 and track_id not in count_ids:
                        count_ids.append(track_id)

                        # Increment total counts
                        total_count += 1
                        class_wise_count[class_names[cls]]["total"] += 1

    # Create a white background for the table with a black border
    table_start_x, table_start_y = 10, 10
    row_height = 30
    col_width = 120
    table_width = 2 * col_width
    table_height = (len(class_wise_count) + 2) * row_height

    # Draw the white background
    cv2.rectangle(im0, (table_start_x, table_start_y),
                  (table_start_x + table_width, table_start_y + table_height),
                  (255, 255, 255), -1)

    # Draw the black border
    cv2.rectangle(im0, (table_start_x, table_start_y),
                  (table_start_x + table_width, table_start_y + table_height),
                  (0, 0, 0), 2)

    # Headers
    cv2.putText(im0, "Vehicles", (table_start_x + 10, table_start_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(im0, "Count", (table_start_x + col_width + 10, table_start_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw horizontal line under headers
    cv2.line(im0, (table_start_x, table_start_y + row_height),
             (table_start_x + table_width, table_start_y + row_height),
             (0, 0, 0), 1)

    # Fill in the class counts
    for idx, (key, value) in enumerate(class_wise_count.items(), start=1):
        y_position = table_start_y + row_height * (idx + 1) - 5
        cv2.putText(im0, str.capitalize(key), (table_start_x + 10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(im0, str(value["total"]), (table_start_x + col_width + 10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw horizontal line above total row
    cv2.line(im0, (table_start_x, table_start_y + row_height * (len(class_wise_count) + 1)),
             (table_start_x + table_width, table_start_y + row_height * (len(class_wise_count) + 1)),
             (0, 0, 0), 1)

    # Total count
    cv2.putText(im0, "Total", (table_start_x + 10, table_start_y + row_height * (len(class_wise_count) + 2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(im0, str(total_count),
                (table_start_x + col_width + 10, table_start_y + row_height * (len(class_wise_count) + 2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("frames", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
