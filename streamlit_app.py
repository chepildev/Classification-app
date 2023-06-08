import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict

st.title("Image Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Load the YOLO model
    model = YOLO("yolov8s.pt")

    # Run YOLOv8 inference on the image
    results = model(image)

    class_counts = defaultdict(int)

    if results:
        # For each result
        for result in results:
            # Get the bounding box coordinates and class ids of detected objects
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # For each detected object
            for box, cls_id in zip(boxes, cls_ids):
                # Convert the class id to integer
                int_id = int(cls_id)

                # Map the class id to the corresponding class name
                class_name = model.names[int_id]

                # Increase the count of the class name
                class_counts[class_name] += 1

            # Get an image with bounding boxes
            annotated_img = result.plot()

            # Display the image with bounding boxes
            st.image(annotated_img, use_column_width=True, channels='BGR')

        # Write the counts of detected objects to the app
        st.subheader('Detected objects:')
        for name, count in class_counts.items():
            st.write(f'{name}: {count}')
