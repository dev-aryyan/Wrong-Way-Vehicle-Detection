import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
video_path = "wrongaireal.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

H, W = frame.shape[:2]
LINE_Y = int(H // 2)

# --- Click to set divider ---
DIVIDER_X = None

def mouse_click(event, x, y, flags, param):
    global DIVIDER_X
    if event == cv2.EVENT_LBUTTONDOWN:
        # only accept clicks near the counting line
        if abs(y - LINE_Y) < 30:
            DIVIDER_X = x
            print(f"Divider set at X = {x}")

# Show first frame and wait for click
setup_frame = frame.copy()
cv2.line(setup_frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
cv2.putText(setup_frame, "Click on the red line to set lane divider, then press ENTER",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

cv2.imshow("Traffic Monitor", setup_frame)
cv2.setMouseCallback("Traffic Monitor", mouse_click)

while True:
    # update marker as user clicks
    preview = setup_frame.copy()
    if DIVIDER_X is not None:
        cv2.circle(preview, (DIVIDER_X, LINE_Y), 8, (0, 255, 0), -1)
        cv2.putText(preview, f"Divider X = {DIVIDER_X}  |  Press ENTER to start",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("Traffic Monitor", preview)
    key = cv2.waitKey(20)
    if key == 13 and DIVIDER_X is not None:  # ENTER
        break
    if key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()

# --- Labels and state ---
labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

track_positions = {}
crossed_ids     = {}
wrong_way_ids   = {}
in_counts       = {2: 0, 3: 0, 5: 0, 7: 0}
out_counts      = {2: 0, 3: 0, 5: 0, 7: 0}


def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def compute_iou(boxA, boxB):
    """Check how much two boxes overlap. Returns 0.0 to 1.0"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def remove_duplicate_boxes(boxes, track_ids, classes, iou_threshold=0.4):
    """Remove duplicate detections of the same large vehicle"""
    keep = []
    suppressed = set()

    for i in range(len(boxes)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed.add(j)  # remove the duplicate

    return (boxes[keep],
            track_ids[keep],
            classes[keep])


# --- Main loop ---
for result in model.track(
    source=video_path,
    stream=True,
    classes=[2, 3, 5, 7],
    persist=True,
    imgsz=640
):
    frame = result.orig_img.copy()

    # Draw counting line and divider marker
    cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
    cv2.circle(frame, (DIVIDER_X, LINE_Y), 6, (0, 255, 0), -1)

    if result.boxes is None or result.boxes.id is None:
        cv2.imshow("Traffic Monitor", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes     = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    classes   = result.boxes.cls.cpu().numpy().astype(int)

    # Remove duplicate overlapping boxes
    boxes, track_ids, classes = remove_duplicate_boxes(boxes, track_ids, classes)

    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        cx, cy = get_center(box)
        x1, y1, x2, y2 = map(int, box)
        cls = classes[i]

        # --- Crossing logic ---
        if tid in track_positions:
            prev_y = track_positions[tid]

            crossed_down = prev_y < LINE_Y <= cy
            crossed_up   = prev_y > LINE_Y >= cy

            if (crossed_down or crossed_up) and tid not in crossed_ids:

                side = "left" if cx < DIVIDER_X else "right"

                if (side == "left" and crossed_up ) or (side == "right" and crossed_down):
                    wrong_way = False
                else:
                    wrong_way = True

                if crossed_down:
                    in_counts[cls] += 1
                    crossed_ids[tid] = "in"
                else:
                    out_counts[cls] += 1
                    crossed_ids[tid] = "out"

                wrong_way_ids[tid] = wrong_way

        track_positions[tid] = cy

        # --- Draw bounding box ---
        if wrong_way_ids.get(tid, False):
            box_color = (0, 0, 255)
            label     = f"ID {tid} WRONG WAY"
        else:
            box_color = (255, 165, 0)
            label     = f"ID {tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # --- Dashboard ---
    cv2.rectangle(frame, (10, 10), (340, 30 + len(labels) * 30), (0, 0, 0), -1)
    y = 35
    for cls_id, name in labels.items():
        cv2.putText(frame, f"{name:<12} In: {in_counts[cls_id]}  Out: {out_counts[cls_id]}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30

    cv2.imshow("Traffic Monitor", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()