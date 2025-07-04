import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
from collections import deque
import traceback

# --- Configuration ---
MODEL_DIR = 'sign_model_focused_enhanced_attention_v2_0.9880_prior1'
MODEL_PATH = os.path.join(MODEL_DIR, 'corrected_enhanced_focused_attention_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SEQUENCE_LENGTH = 32
NUM_SELECTED_POSE_LANDMARKS = 7
UPPER_BODY_POSE_LANDMARKS_INDICES = [0, 11, 12, 13, 14, 15, 16]
PREDICTION_THRESHOLD = 0.60
PREDICTION_BUFFER_SIZE = 10

@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class AttentionWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('A list of two tensors is expected. Got: {}'.format(inputs))
        gru_output, attention_probs = inputs
        weighted_sequence = gru_output * attention_probs
        context_vector = tf.reduce_sum(weighted_sequence, axis=1)
        return context_vector
    def get_config(self):
        base_config = super().get_config()
        return base_config

def load_model_with_custom_objects(model_path):
    custom_objects = {'AttentionWeightedAverage': AttentionWeightedAverage}
    try:
        print(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model recompiled.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def extract_focused_keypoints_realtime(results):
    pose_keypoints_list = []
    if results.pose_landmarks:
        for idx in UPPER_BODY_POSE_LANDMARKS_INDICES:
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                pose_keypoints_list.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                pose_keypoints_list.extend([0,0,0,0])
    if len(pose_keypoints_list) < NUM_SELECTED_POSE_LANDMARKS * 4:
        pose_keypoints_list.extend([0] * (NUM_SELECTED_POSE_LANDMARKS * 4 - len(pose_keypoints_list)))
    pose = np.array(pose_keypoints_list).flatten()
    if pose.size == 0:
        pose = np.zeros(NUM_SELECTED_POSE_LANDMARKS * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def add_temporal_features_realtime(X_scaled):
    if X_scaled.shape[0] < 2:
        X_diff1 = np.zeros_like(X_scaled)
        X_diff2 = np.zeros_like(X_scaled)
    else:
        X_diff1 = np.diff(X_scaled, axis=0, prepend=X_scaled[:1, :])
        X_diff2 = np.diff(X_diff1, axis=0, prepend=X_diff1[:1, :])
    velocity_mag = np.linalg.norm(X_diff1, axis=-1, keepdims=True)
    acceleration_mag = np.linalg.norm(X_diff2, axis=-1, keepdims=True)
    return np.concatenate([X_scaled, X_diff1, X_diff2, velocity_mag, acceleration_mag], axis=-1)

def predict_on_video(video_path, output_path=None):
    print(f"Loading resources for ENHANCED FOCUSED model from: {MODEL_DIR}")
    model = load_model_with_custom_objects(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    if not all([model, scaler, label_encoder]):
        print("Failed to load one or more resources. Exiting.")
        return
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    print(f"Actions: {actions_map}")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        holistic.close()
        return
    sequence_data_raw = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    current_prediction_text = "..."
    current_confidence = 0.0
    frame_count = 0
    results_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        keypoints = extract_focused_keypoints_realtime(results)
        sequence_data_raw.append(keypoints)
        if len(sequence_data_raw) == SEQUENCE_LENGTH:
            try:
                X_seq_raw = np.array(sequence_data_raw)
                original_shape = X_seq_raw.shape
                X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
                X_scaled = scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(original_shape)
                X_enhanced = add_temporal_features_realtime(X_scaled)
                X_input = np.expand_dims(X_enhanced, axis=0)
                prediction_probabilities = model.predict(X_input)[0]
                predicted_index = np.argmax(prediction_probabilities)
                confidence = prediction_probabilities[predicted_index]
                if confidence > PREDICTION_THRESHOLD:
                    prediction_buffer.append(predicted_index)
                    current_confidence = confidence
                if len(prediction_buffer) > 0:
                    most_common_pred_index = max(set(prediction_buffer), key=prediction_buffer.count)
                    current_prediction_text = actions_map.get(most_common_pred_index, "...")
                    if most_common_pred_index == predicted_index and confidence > PREDICTION_THRESHOLD:
                        current_confidence = confidence
                else:
                    current_prediction_text = "..."
                    current_confidence = 0.0
            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()
                current_prediction_text = "Error"
                current_confidence = 0.0
        results_list.append({
            'frame': frame_count,
            'prediction': current_prediction_text,
            'confidence': float(current_confidence)
        })
        frame_count += 1
        if output_path:
            # Optionally, draw and save annotated video
            annotated_frame = frame.copy()
            try:
                mp_drawing = mp.solutions.drawing_utils
                mp_holistic = mp.solutions.holistic
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_hands = mp.solutions.hands
                mp_drawing.draw_landmarks(
                    annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    annotated_frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    annotated_frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
                text_to_display = f"Prediction: {current_prediction_text.upper()} ({current_confidence:.2f})"
                cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], 40), (245, 117, 16), -1)
                cv2.putText(annotated_frame, text_to_display, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if frame_count == 1:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (annotated_frame.shape[1], annotated_frame.shape[0]))
                out.write(annotated_frame)
            except Exception as e:
                print(f"Error drawing/annotating frame: {e}")
    cap.release()
    holistic.close()
    if output_path and 'out' in locals():
        out.release()
    print("Prediction on video completed.")
    return results_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run sign prediction on a video file.")
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Optional path to save annotated output video')
    args = parser.parse_args()
    results = predict_on_video(args.video_path, args.output)
    print("Frame-wise predictions:")
    for r in results:
        print(r)
