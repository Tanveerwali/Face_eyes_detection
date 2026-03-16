import cv2
import os
from datetime import datetime

output_folder = "../data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_folder, f"face_eye_video_{timestamp}.avi")

face_cascade = cv2.CascadeClassifier("C:\\Users\\Shekhani Laptops\\OneDrive\\Desktop\\HaarCascadeProject\\cascades\\haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Users\Shekhani Laptops\\OneDrive\\Desktop\\HaarCascadeProject\\cascades\\haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

print(f"Recording video to: {output_path}")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(60, 60)
    )

    print(f"Faces detected: {len(faces)}")  

    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    
    cv2.imshow("Face & Eye Detection", frame)

    
    out.write(frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully!")