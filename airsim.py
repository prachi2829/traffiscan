from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2, os, time, numpy as np
import matplotlib.pyplot as plt

input_video = "videos/cardrive.mp4"
print(f"Using simulator video: {input_video}")


model = YOLO("models/best.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    embedder="mobilenet"
)


cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file: {input_video}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("assets/airsim_track", exist_ok=True)
out_path = "assets/airsim_track/output_deepsort.mp4"
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
id_switches = 0
detections_count = 0
prev_ids = set()
start_time = time.time()
tracked_lengths = {}

print("\nStarting YOLOv8 + DeepSORT Tracking... Press Q to quit.\n")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        detections_count += 1

    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        current_ids.add(tid)
        tracked_lengths[tid] = tracked_lengths.get(tid, 0) + 1

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    disappeared = prev_ids - current_ids
    appeared = current_ids - prev_ids
    if disappeared and appeared:
        id_switches += min(len(appeared), len(disappeared))
    prev_ids = current_ids

    out.write(frame)
    cv2.imshow("YOLOv8 + DeepSORT (Simulator Video)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


elapsed = time.time() - start_time
fps_actual = frame_count / elapsed
avg_track_len = np.mean(list(tracked_lengths.values())) if tracked_lengths else 0
mota = max(0, 1 - (id_switches / max(detections_count, 1)))

#Approximate MOTA computed using ID-switch statistics in the absence of ground-truth tracking labels.

print("\n========== TRACKING PERFORMANCE ==========")
print(f"Frames Processed     : {frame_count}")
print(f"Detections Count     : {detections_count}")
print(f"Average FPS          : {fps_actual:.2f}")
print(f"ID Switches          : {id_switches}")
print(f"MOTA (Accuracy)      : {mota:.3f}")
print(f"Avg Track Length     : {avg_track_len:.2f} frames")
print(f"Output Video Saved   : {out_path}")
print("==========================================\n")


metrics = ["MOTA", "FPS", "ID Switches", "Avg Track Len"]
values = [mota,  fps_actual, id_switches, avg_track_len]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics, values)
plt.title("TraffiScan Tracking Performance Metrics")
plt.ylabel("Values")

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/4, bar.get_height() + 0.02,
             f"{values[i]:.3f}", fontsize=9)

plt.tight_layout()
graph_path = "assets/airsim_track/tracking_metrics_graph1.png"
plt.savefig(graph_path)
plt.show()

print(f"Tracking graph saved at: {graph_path}")
print("Done! TraffiScan tracking with evaluation completed.")
