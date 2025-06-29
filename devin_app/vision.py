"""Webcam utilities for capturing images and displaying a live feed."""
import threading
import cv2
import numpy as np
from PIL import ImageGrab
import logging

from .config import WINDOW_NAME
CAMERA_INDEX = 0

camera = None
camera_lock = threading.Lock()


def initialize_webcam() -> bool:
    global camera
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        logging.error("Error: Camera not initialized.")
        return False
    if not camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920):
        logging.warning("Failed to set frame width to 1920.")
    if not camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080):
        logging.warning("Failed to set frame height to 1080.")
    logging.info("Webcam initialized successfully.")
    return True


def display_live_feed(stop_signal: threading.Event):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while not stop_signal.is_set():
        with camera_lock:
            ret, frame = camera.read()
            if not ret:
                logging.error("Error: Frame capture failed.")
                break
            frame_filtered = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            cv2.imshow(WINDOW_NAME, frame_filtered)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_signal.set()
            break
    with camera_lock:
        camera.release()
    cv2.destroyAllWindows()
    logging.info("Live feed stopped and camera released.")


def capture_image_from_webcam() -> dict:
    global camera
    with camera_lock:
        ret, frame = camera.read()
        if not ret:
            logging.error("Error: Failed to capture image from webcam.")
            return {}
        frame_filtered = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        ret_jpg, buffer = cv2.imencode(".jpg", frame_filtered, encode_param)
        if not ret_jpg:
            logging.error("Error: Failed to encode webcam image.")
            return {}
    screenshot = ImageGrab.grab()
    screenshot_rgb = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)
    screenshot_filtered = cv2.bilateralFilter(screenshot_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    ret_ss_jpg, screenshot_buffer = cv2.imencode(".jpg", screenshot_filtered, encode_param)
    if not ret_ss_jpg:
        logging.error("Error: Failed to encode screenshot.")
        return {}
    return {
        "webcam_image": {"mime_type": "image/jpeg", "data": buffer.tobytes()},
        "screenshot": {"mime_type": "image/jpeg", "data": screenshot_buffer.tobytes()},
    }
