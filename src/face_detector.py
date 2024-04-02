import cv2
import os

def face_extraction(input_video_path, output_folder):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Save each detected face as PNG
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f"face_{frame_count}_.png"), face)
        
        # Display the frame with rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\output\video\test_video01_subtitled.mp4'
    outout_folder = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\output\faces'
    face_extraction(input_video_path, outout_folder)