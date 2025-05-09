#import libraies
# Make sure you installed 
import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
import time
from gradio_client import Client, handle_file


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#Function for calculate the agles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle#Detect only the inner angles


def detect_lunge(left_knee_angle, right_knee_angle, leg_angle):
    if 80 < left_knee_angle < 110 and 80 < right_knee_angle < 110:#Determine if the angles are correct
        if 80 < leg_angle < 110:
            return "Lunge Detected: Correct", (0, 255, 0)#Green
        else:
            return "Lunge Detected but Legs Not Open Properly", (255, 255, 0)#Yellow
    return "Incorrect Lunge or No Lunge", (255, 0, 0 )# Red in RGB formats

def detect_sit_up(left_knee_angle, right_knee_angle): 
    if left_knee_angle < 70 and right_knee_angle < 70:
        return "Sit up Detected: Correct", (0, 255, 0)
    return "Incorrect Sit Up or No Sit Up", (255, 0, 0)


def process_webcam(pose_id):
    cap = cv2.VideoCapture(0)#use 0 for a webcam stream
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = pose.process(image)

        pose_text = "No Pose Detected"#Default as no pose
        color = (255, 255, 255)# in white colour
        image_height, image_width, _ = image.shape#Define the image size

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            #capture the key point locations according to the frame size

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height]
            
            # Calculate knee angles.
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            if pose_id == 1:
                # Calculate the mid-hip point.
                mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
                leg_angle = calculate_angle(left_knee, mid_hip, right_knee)
                pose_text, color = detect_lunge(left_knee_angle, right_knee_angle, leg_angle)
            elif pose_id == 2:
                pose_text, color = detect_sit_up(left_knee_angle, right_knee_angle)

            cv2.putText(image, pose_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        yield image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#Change the color back to RGB to make it seems more normal
        time.sleep(0.03)

    cap.release()
    pose.close()

#Call API to get prediction data
def predict_from_file(pose_id, video_file):
    client = Client("https://f91089146d194cfae2.gradio.live/")
    result = client.predict(
        pose_id=str(pose_id),
        video_file=video_file,
        api_name="/predict"
    )
    return result[0], result[1]

#the UI
with gr.Blocks(title="Pose Detection App") as app:
    gr.Markdown("## Real-Time or Uploaded Video Pose Detection")
    with gr.Tab("Live Webcam"):
        pose_select_live = gr.Radio([("Lunge", 1), ("Sit-up", 2)], label="Select Exercise", value=1)
        webcam_display = gr.Image()
        start_btn = gr.Button("Start Webcam")
        start_btn.click(fn=process_webcam, inputs=pose_select_live, outputs=webcam_display)

    with gr.Tab("Upload Video via API"):
        pose_select_upload = gr.Radio([("Lunge", 1), ("Sit-up", 2)], label="Select Pose", value=1)
        video_file_input = gr.File(label="Upload Video File")
        video_result = gr.Video(label="Processed Video")
        html_output = gr.HTML(label="Download Button")
        predict_btn = gr.Button("Run Detection")
        predict_btn.click(fn=predict_from_file, inputs=[pose_select_upload, video_file_input], outputs=[video_result, html_output])

#launch the app and create local and public URL for prediction
app.launch(server_name="0.0.0.0", server_port=7860, share=True)
