import cv2
import pandas as pd
from datetime import datetime
import os

class MotionDetector:
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
        self.first_frame = None
        self.status_list = []
        self.detections = []
        self.df = pd.DataFrame(columns=["Date", "Time"])
        self.snapshot_dir = "snapshots"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        self.video.release()

    def detect_motion(self):
        video_writer = None

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script directory
        snapshot_dir = os.path.join(script_dir, self.snapshot_dir)  # Create the snapshot directory path
        os.makedirs(snapshot_dir, exist_ok=True)  # Create the "snapshots" folder if it doesn't exist

        while True:
            check, frame = self.video.read()
            if not check:
                break

            status = -1
            text = 'Unoccupied'

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

            if self.first_frame is None:
                self.first_frame = gray_img
                continue

            delta_frame = cv2.absdiff(self.first_frame, gray_img)
            thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False  # Flag to track motion detection

            for contour in contours:
                if cv2.contourArea(contour) < 10000:
                    continue
                status = 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                text = 'Occupied'

                # Face detection
                gray_roi = gray_img[y:y+h, x:x+w]
                faces = self.face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        # Draw rectangle around the face
                        cv2.rectangle(frame, (x+fx, y+fy), (x+fx+fw, y+fy+fh), (255, 0, 0), 2)
                        # Save snapshot
                        snapshot_filename = f"snapshot_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                        snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                        cv2.imwrite(snapshot_path, frame)
                        print(f"Snapshot taken: {snapshot_path}")
                        motion_detected = True

                continue

            cv2.putText(frame, '[Room Status]: %s' % text,
                        (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.50, (255, 255, 255), 1)

            current_time = datetime.now()
            timestamp = current_time.strftime('%d %B %Y, %A %I:%M:%S %p')  # Get current date, day, and time in 12-hour format

            cv2.putText(frame, timestamp,
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 255, 255), 1)

            self.status_list.append(status)

            if len(self.status_list) >= 2 and (self.status_list[-1] * self.status_list[-2]) == -1 and motion_detected:
                motion_time = datetime.now()
                self.detections.append((motion_time.date(), motion_time.strftime('%I:%M:%S %p')))

            cv2.imshow("Colour Frame", frame)

            if video_writer is None:
                fps = int(self.video.get(cv2.CAP_PROP_FPS))
                width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter('Feed.avi',
                                               cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                                               fps, (width, height))
            video_writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        if video_writer is not None:
            video_writer.release()

    def save_detections_to_csv(self, filename):
        if len(self.detections) > 0:
            df = pd.DataFrame(self.detections, columns=["Date", "Time"])
            df.to_csv(filename, index=False)


# Usage
video_source = 0  # Replace with video file path if using a file as input
output_filename = "MotionDetections.csv"

with MotionDetector(video_source) as detector:
    detector.detect_motion()
