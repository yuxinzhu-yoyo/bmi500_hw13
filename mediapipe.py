import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

inputvid = cv2.VideoCapture("/Users/yzhu595/input.mp4")


frame_width = int(inputvid.get(3))
frame_height = int(inputvid.get(4))
fps = int(inputvid.get(5))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputvid = cv2.VideoWriter('/Users/yzhu595/output.mp4', fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:
    
    while True:
        success, image = inputvid.read()

        if not success:
            break

        image.flags.writeable = False
        #BGR to RGB conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image) #mp run

        image.flags.writeable = True
         #RGB to BGR conversion
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # output file
        outputvid.write(image)

        if cv2.waitKey(20) == ord('q'):
            break
inputvid.release()
cv2.destroyAllWindows()
