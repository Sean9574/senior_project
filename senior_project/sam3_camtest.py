#!/usr/bin/env python3
"""SAM3 Webcam - Adjustable resolution"""

import base64
import threading
import time

import cv2
import numpy as np
import requests

SERVER = "http://localhost:8100"
FLIP_CAMERA = True

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

class SAM3Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(4)
        self.current_frame = None
        self.result = {"prompt": "loading...", "num_objects": 0, "boxes": [], "scores": [], "masks_base64": []}
        self.running = True
        self.lock = threading.Lock()
        self.inference_fps = 0
        
        # Adjustable settings
        self.frame_width = 800
        self.confidence = 0.2
        
    def decode_mask(self, mask_b64):
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            return mask
        except:
            return None
    
    def overlay_masks(self, frame, masks_b64, boxes, scores, prompt):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        for i, (mask_b64, box, score) in enumerate(zip(masks_b64, boxes, scores)):
            mask = self.decode_mask(mask_b64)
            if mask is None:
                continue
            
            mask = cv2.resize(mask, (w, h))
            color = COLORS[i % len(COLORS)]
            
            mask_bool = mask > 127
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"{prompt}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
        
    def capture_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if FLIP_CAMERA:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (self.frame_width, int(h * self.frame_width / w)))
                with self.lock:
                    self.current_frame = small.copy()
            time.sleep(0.03)
    
    def inference_thread(self):
        while self.running:
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
                conf = self.confidence
            
            try:
                start = time.time()
                
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_b64 = base64.b64encode(buf).decode()
                
                r = requests.post(f"{SERVER}/segment",
                    json={"image_base64": img_b64, "confidence_threshold": conf},
                    timeout=10)
                
                self.inference_fps = 1.0 / (time.time() - start)
                
                with self.lock:
                    self.result = r.json()
                    
            except Exception as e:
                print(f"Inference error: {e}")
            
            time.sleep(0.01)
    
    def run(self):
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.inference_thread, daemon=True).start()
        
        print("Controls:")
        print("  q - quit")
        print("  f - flip camera")
        print("  +/- - adjust resolution")
        print("  [/] - adjust confidence")
        
        global FLIP_CAMERA
        
        while self.running:
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
                result = self.result.copy()
                fps = self.inference_fps
            
            prompt = result.get("prompt", "?")
            num = result.get("num_objects", 0)
            masks_b64 = result.get("masks_base64", [])
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            
            if masks_b64 and boxes:
                display = self.overlay_masks(frame, masks_b64, boxes, scores, prompt)
            else:
                display = frame
            
            # Info bar
            cv2.putText(display, f"'{prompt}' | Found: {num} | {fps:.1f}Hz | Res:{self.frame_width} | Conf:{self.confidence:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display, "q:quit f:flip +/-:res [/]:conf",
                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            cv2.imshow("SAM3 Segmentation", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('f'):
                FLIP_CAMERA = not FLIP_CAMERA
            elif key == ord('+') or key == ord('='):
                self.frame_width = min(800, self.frame_width + 80)
                print(f"Resolution: {self.frame_width}")
            elif key == ord('-'):
                self.frame_width = max(240, self.frame_width - 80)
                print(f"Resolution: {self.frame_width}")
            elif key == ord(']'):
                self.confidence = min(0.9, self.confidence + 0.05)
                print(f"Confidence: {self.confidence:.2f}")
            elif key == ord('['):
                self.confidence = max(0.05, self.confidence - 0.05)
                print(f"Confidence: {self.confidence:.2f}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SAM3Webcam().run()