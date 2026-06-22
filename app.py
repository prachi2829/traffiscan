
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(page_title="TraffiScan", layout="wide")

st.markdown("""
<div style='text-align:center'>
    <h1> TraffiScan</h1>
    <h3>YOLOv8 + DeepSORT Object Detection and Tracking</h3>
</div>
<hr>
""", unsafe_allow_html=True)

st.sidebar.title("Settings")

model_path = st.sidebar.text_input("Model", "models/best.pt")

conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25
)

mode = st.sidebar.radio(
    "Mode",
    ["YOLO Detection", "YOLO + DeepSORT Tracking"]
)

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Original Video")
        st.video(uploaded_video.getvalue())

    process_btn = st.button("Process Video")

    if process_btn:

        model = YOLO(model_path)

        tracker = None
        if mode == "YOLO + DeepSORT Tracking":
            tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4
            )

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )
        temp_input.write(uploaded_video.getvalue())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        

        progress = st.progress(0)

        with col2:
            st.subheader("Live Processing")
            stframe = st.empty()

        last_frame = None

        frame_count = 0
        detections_count = 0
        id_switches = 0

        tracked_ids = set()
        tracked_lengths = {}
        prev_ids = set()

        start_time = time.time()

        with st.spinner("Processing Video..."):

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                results = model(
                    frame,
                    conf=conf_thresh,
                    verbose=False
                )[0]

                if mode == "YOLO Detection":

                    detections_count += len(results.boxes)

                    annotated = results.plot()

                else:

                    detections = []
                    current_ids = set()

                    for box in results.boxes:

                        x1, y1, x2, y2 = box.xyxy[0]

                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        detections.append([
                            [
                                int(x1),
                                int(y1),
                                int(x2 - x1),
                                int(y2 - y1)
                            ],
                            conf,
                            cls
                        ])

                        detections_count += 1

                    tracks = tracker.update_tracks(
                        detections,
                        frame=frame
                    )

                    for track in tracks:

                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id

                        tracked_ids.add(track_id)
                        current_ids.add(track_id)

                        tracked_lengths[track_id] = (
                            tracked_lengths.get(track_id, 0) + 1
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            track.to_ltrb()
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"ID {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                    disappeared = prev_ids - current_ids
                    appeared = current_ids - prev_ids

                    if disappeared and appeared:
                        id_switches += min(
                            len(disappeared),
                            len(appeared)
                        )

                    prev_ids = current_ids

                    annotated = frame

                annotated = cv2.resize(
                    annotated,
                    (width, height)
                )

                last_frame = annotated.copy()

                stframe.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )

                

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

        cap.release()
        

        elapsed = time.time() - start_time

        fps_actual = frame_count / elapsed if elapsed > 0 else 0

        avg_track_len = (
            np.mean(list(tracked_lengths.values()))
            if tracked_lengths else 0
        )

        mota = max(
            0,
            1 - (
                id_switches /
                max(detections_count, 1)
            )
        )

        


        progress.empty()
        stframe.empty()

        with col2:
            st.subheader("Final Result")

            if last_frame is not None:
                st.image(
                    last_frame,
                    channels="BGR",
                    caption="Final Processed Frame",
                    use_container_width=True
                )

            st.success("Processing Complete")

        st.markdown("## Performance Metrics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("FPS", f"{fps_actual:.2f}")
        c2.metric("Frames", frame_count)
        c3.metric("Detections", detections_count)
        c4.metric("Track IDs", len(tracked_ids))

        if mode == "YOLO + DeepSORT Tracking":

            st.markdown("##  Tracking Analytics")

            t1, t2, t3 = st.columns(3)

            t1.metric("MOTA", f"{mota:.3f}")
            t2.metric("ID Switches", id_switches)
            t3.metric("Avg Track Length", f"{avg_track_len:.1f}")

           

            metrics = [
                "MOTA",
                "FPS",
                "ID Switches",
                "Avg Track Len"
            ]

            values = [
                mota,
                fps_actual,
                id_switches,
                avg_track_len
            ]

            fig, ax = plt.subplots(figsize=(8, 5))

            bars = ax.bar(metrics, values)

            for bar in bars:
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f"{height:.3f}",
                    ha="center"
                )

            ax.set_title(
                "TraffiScan Tracking Performance"
            )

            st.pyplot(fig)

st.markdown("---")
st.markdown(
    "<center>TraffiScan | YOLOv8 + DeepSORT + Streamlit</center>",
    unsafe_allow_html=True
)
