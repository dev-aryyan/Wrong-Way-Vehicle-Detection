import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
video_path = "sample_video.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W = frame.shape[:2]
LINE_Y = int(H // 2)
cap.release()

# --- Click to set divider ---
DIVIDER_X = None

def mouse_click(event, x, y, flags, param):
    global DIVIDER_X
    if event == cv2.EVENT_LBUTTONDOWN and abs(y - LINE_Y) < 30:
        DIVIDER_X = x

setup_frame = frame.copy()
cv2.line(setup_frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
cv2.putText(setup_frame, "Click red line to set divider, then press ENTER",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

cv2.imshow("Traffic Monitor", setup_frame)
cv2.setMouseCallback("Traffic Monitor", mouse_click)

while True:
    preview = setup_frame.copy()
    if DIVIDER_X is not None:
        cv2.circle(preview, (DIVIDER_X, LINE_Y), 8, (0, 255, 0), -1)
    cv2.imshow("Traffic Monitor", preview)
    key = cv2.waitKey(20)
    if key == 13 and DIVIDER_X is not None:
        break
    if key == 27:
        cv2.destroyAllWindows()
        exit()

# --- Tracking state ---
track_positions = {}
crossed_ids     = {}
wrong_way_ids   = {}

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# --- Main loop ---
for result in model.track(
    source=video_path,
    stream=True,
    classes=[2, 3, 5, 7],
    persist=True,
    imgsz=640
):
    frame = result.orig_img.copy()
    cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
    cv2.circle(frame, (DIVIDER_X, LINE_Y), 6, (0, 255, 0), -1)

    if result.boxes is None or result.boxes.id is None:
        cv2.imshow("Traffic Monitor", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes     = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, tid in zip(boxes, track_ids):
        cx, cy = get_center(box)
        x1, y1, x2, y2 = map(int, box)

        # --- Wrong way check at crossing moment ---
        if tid in track_positions:
            prev_y = track_positions[tid]
            crossed_down = prev_y < LINE_Y <= cy
            crossed_up   = prev_y > LINE_Y >= cy

            if (crossed_down or crossed_up) and tid not in crossed_ids:
                side = "left" if cx < DIVIDER_X else "right"

                if (side == "left" and crossed_up) or (side == "right" and crossed_down):
                    wrong_way_ids[tid] = False
                else:
                    wrong_way_ids[tid] = True

                crossed_ids[tid] = True

        track_positions[tid] = cy

        # --- Draw ---
        if wrong_way_ids.get(tid, False):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID {tid} WRONG WAY", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            

    cv2.imshow("Traffic Monitor", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
