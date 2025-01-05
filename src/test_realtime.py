import numpy as np
from scipy.special import expit
import cv2
import mediapipe as mp
import joblib
import warnings
import keras
from threading import Thread, Lock
import time
warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
emotion_model = keras.models.load_model('models/facial_emotion_classifier.h5')
pca_model = joblib.load('models/pca_model.pkl')
EMOTION_LABELS = ['angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']

cap = None
cap_id = 0
results_mesh = None
mesh_points = None
emotion_scores = [0] * len(EMOTION_LABELS)
processed_window = []
window_lock = Lock()

def open_camera(camera_id):
    global cap
    while True:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Camera failure")
        else:
            return cap

def transform_to_zero_one_numpy(arr, normalize=False):
    if not len(arr):
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr)
    value_range = max_val - min_val
    transformed_arr = (arr - min_val) / value_range
    if normalize:
        transformed_arr /= np.sum(transformed_arr)
    return transformed_arr

def update_mesh_points(frame):
    global results_mesh, mesh_points
    H, W, _ = frame.shape
    results_mesh = face_mesh.process(frame)
    if results_mesh.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y, p.z], [W, H, max(W, H)]).astype(int) for p in results_mesh.multi_face_landmarks[0].landmark])

def draw_tracking(frame):
    global results_mesh, mesh_points
    if results_mesh.multi_face_landmarks is not None:
        for faceLms in results_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, faceLms, mp_face_mesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
    return frame

def draw_emotion(frame, emotion):
    if emotion:
        cv2.putText(frame, f"Expr: {emotion.capitalize()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def predict_emotion(frame, draw=False):
    global mesh_points, processed_window
    if mesh_points is None:
        return None
    nose_tip = mesh_points[4]
    forehead = mesh_points[151]
    mesh_norm = mesh_points - nose_tip
    scale_factor = np.linalg.norm(forehead - nose_tip)
    if np.isclose(scale_factor, 0):
        scale_factor = 1e-6
    mesh_norm = np.divide(mesh_norm, scale_factor)
    landmarks_flat = mesh_norm.flatten()
    landmarks_transformed = pca_model.transform([landmarks_flat])
    with window_lock:
        processed_window.append(landmarks_transformed[0])
    if draw:
        frame = draw_emotion(frame, EMOTION_LABELS[np.argmax(emotion_scores)])
    return frame

def prediction_thread():
    global processed_window, emotion_scores
    while True:
        with window_lock:
            if len(processed_window) >= 30:
                batch = np.array([processed_window[-30:]])
                pred = emotion_model.predict(batch, verbose=0)[0]
                emotion_scores_noisy = transform_to_zero_one_numpy(pred)
                print(emotion_scores_noisy)
                for score in range(len(emotion_scores)):
                    emotion_scores_noisy[score] = expit(10 * (emotion_scores_noisy[score] - 0.5))
                    emotion_scores[score] = emotion_scores[score]*0.5 + emotion_scores_noisy[score]*0.5
                processed_window = []
        time.sleep(0.1)

def main(draw=False):
    ret, frame = cap.read()
    if frame is not None:
        update_mesh_points(frame)
        frame = predict_emotion(frame, draw=draw)
        if draw:
            try:
                frame = draw_tracking(frame)
                cv2.imshow('frame', frame)
            except cv2.error:
                print("Frame not ready")
        cv2.waitKey(1)
        return frame
    else:
        open_camera(cap_id)
        return None

if __name__ == "__main__":
    open_camera(cap_id)
    pred_thread = Thread(target=prediction_thread)
    pred_thread.start()
    fps = 0
    try:
        while True:
            start = time.time()
            frame = main(draw=True)
            fps = (fps*0.9) + ((1/(time.time()-start))*0.1)
            #print(fps)
    except KeyboardInterrupt:
        pred_thread.join()
        cap.release()
        cv2.destroyAllWindows()