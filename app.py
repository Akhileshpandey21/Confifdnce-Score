import cv2
import mediapipe as mp
from deepface import DeepFace
from textblob import TextBlob
import csv
from datetime import datetime
import os

# 
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
# 

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)
# Load Sentence-BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1

class FusionNetwork(nn.Module):

    def __init__(self):

        super(FusionNetwork, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 8)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
    
fusion_model = FusionNetwork()
fusion_model.load_state_dict(
    torch.load("fusion_model.pth")
)

fusion_model.eval()
# def analyze_text_confidence(text):

#     confident_words = [
#         "confident",
#         "successfully",
#         "achieved",
#         "completed",
#         "strong",
#         "excellent",
#         "capable",
#         "skilled"
#     ]

#     nervous_words = [
#         "maybe",
#         "try",
#         "probably",
#         "guess",
#         "not sure",
#         "perhaps",
#         "might"
#     ]

#     text_lower = text.lower()

#     confidence_score = 50

#     # Positive confidence words
#     for word in confident_words:
#         if word in text_lower:
#             confidence_score += 10

#     # Nervous/hesitation words
#     for word in nervous_words:
#         if word in text_lower:
#             confidence_score -= 10

#     # Sentiment analysis
#     sentiment = TextBlob(text).sentiment.polarity

#     confidence_score += sentiment * 20

#     # Clamp score
#     confidence_score = max(0, min(100, confidence_score))

#     return int(confidence_score)

def analyze_text_confidence(text):

    embedding = bert_model.encode(text)

    positive_words = [
        "confident",
        "successfully",
        "achieved",
        "completed",
        "excellent",
        "strong",
        "capable"
    ]

    negative_words = [
        "maybe",
        "try",
        "not sure",
        "guess",
        "perhaps"
    ]

    text_lower = text.lower()

    score = 50

    for word in positive_words:
        if word in text_lower:
            score += 8

    for word in negative_words:
        if word in text_lower:
            score -= 8

    score = max(0, min(100, score))

    return score, embedding


# CSV file name
csv_file = "confidence_logs.csv"

# Create CSV file with header if not exists
if not os.path.exists(csv_file):

    with open(csv_file, mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "User_Text",
            "Emotion",
            "Confidence_Score"
        ])
# Open Webcam
cap = cv2.VideoCapture(0)

previous_nose_x = None
movement_score = 0


frame_count = 0
save_interval = 60
confidence_history = deque(maxlen=50)

user_text = input("Enter your response: ")
# text_confidence = analyze_text_confidence(user_text)
text_confidence, text_embedding = analyze_text_confidence(user_text)

print("Text Confidence Score:", text_confidence)
while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_frame)

    # Draw landmarks
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )
            landmarks = face_landmarks.landmark

            # Get positions
            left_eye_x = landmarks[LEFT_EYE].x
            right_eye_x = landmarks[RIGHT_EYE].x
            nose_x = landmarks[NOSE_TIP].x

            # Eye center
            eye_center = (left_eye_x + right_eye_x) / 2

            # Eye contact detection
            difference = abs(nose_x - eye_center)

            if difference < 0.02:
                eye_contact = "Good"
                eye_score = 100
            else:
                eye_contact = "Weak"
                eye_score = 50

            # Head movement tracking
            if previous_nose_x is not None:

                movement = abs(nose_x - previous_nose_x)
                movement_score += movement

            previous_nose_x = nose_x
            frame_count += 1

    # Emotion Detection
    try:

        emotion_result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        dominant_emotion = emotion_result[0]['dominant_emotion']
                # Emotion scoring
        if dominant_emotion in ['happy', 'neutral']:
            emotion_score = 100
        else:
            emotion_score = 60

        # Stability score
        if frame_count > 0:
            avg_movement = movement_score / frame_count
        else:
            avg_movement = 0

        if avg_movement < 0.01:
            stability_score = 100
        else:
            stability_score = 60

        # Final confidence score
        video_confidence = (
    0.4 * eye_score +
    0.3 * stability_score +
    0.3 * emotion_score
)
    #     confidence_score = (
            
    #         # 0.4 * eye_score +
    #         # 0.3 * stability_score +
    #         # 0.3 * emotion_score
    #         0.7 * video_confidence +
    # 0.3 * text_confidence
    #     )

        fusion_input = torch.tensor(
    [[
        video_confidence / 100,
        text_confidence / 100
    ]],
    dtype=torch.float32
)
        # confidence_score = fusion_model(fusion_input)

        # confidence_score = confidence_score.item()

        # confidence_score = max(0, min(100, confidence_score))
        confidence_score = fusion_model(fusion_input)

# Convert tensor to number
        confidence_score = confidence_score.item()

# Since model trained on normalized values,
# convert back to percentage
        confidence_score = confidence_score * 100

# Clamp values
        confidence_score = max(0, min(100, confidence_score))


        
        confidence_history.append(confidence_score)

        # confidence_score = int(confidence_score)
        confidence_score = round(confidence_score, 2)

        # Display emotion
        cv2.putText(
            frame,
            f'Emotion: {dominant_emotion}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
            
        )
        cv2.putText(
            frame,
            f'Confidence Score: {confidence_score}%',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f'Eye Contact: {eye_contact}',
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f'Text Confidence: {text_confidence}%',
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
                # Actionable feedback

        if confidence_score > 75:
            feedback = "Excellent confidence"
        elif confidence_score > 50:
            feedback = "Moderate confidence"
        else:
            feedback = "Improve eye contact"

        cv2.putText(
            frame,
            f'Feedback: {feedback}',
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
                # Save session data to CSV

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

               # Save every 60 frames
        if frame_count % save_interval == 0:

            with open(csv_file,
                      mode='a',
                      newline='') as file:

                writer = csv.writer(file)

                writer.writerow([
                    current_time,
                    user_text,
                    dominant_emotion,
                    confidence_score
                ])

    except Exception as e:
        print("Error:", e)

    # Show window
    cv2.imshow("AI Confidence & Emotion Detection", frame)

    # Exit with Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Close when window cross button is clicked
    if cv2.getWindowProperty(
    "AI Confidence & Emotion Detection",
    cv2.WND_PROP_VISIBLE
) < 1:
        break

# Release resources

plt.plot(confidence_history)

plt.title("Confidence Over Time")
plt.xlabel("Frames")
plt.ylabel("Confidence Score")
plt.savefig("confidence_graph.png")

plt.show()

cap.release()
cv2.destroyAllWindows()