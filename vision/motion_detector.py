import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=5000, stabilize_frames=10):
        self.threshold = threshold
        self.min_area = min_area
        self.stabilize_frames = stabilize_frames
        
        self.prev_gray = None
        self.stable_counter = 0
        self.was_moving = False
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        moving_area = cv2.countNonZero(thresh)
        is_currently_moving = moving_area > self.min_area

        event_triggered = False

        if is_currently_moving:
            self.was_moving = True
            self.stable_counter = 0
        else:
            if self.was_moving:
                self.stable_counter += 1
                if self.stable_counter >= self.stabilize_frames:
                    event_triggered = True
                    self.was_moving = False
                    self.stable_counter = 0
                    
        self.prev_gray = gray
        return event_triggered
