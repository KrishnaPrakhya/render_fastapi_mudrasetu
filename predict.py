# %% [markdown]
# ## 3) Inference (Enhanced Focused Data with Attention, UI, and Confidence)

# %%
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import os
from collections import deque
import traceback

# --- Configuration ---
MODEL_DIR = 'sign_model_focused_enhanced_attention_v2_0.9880_prior1' # *** Path to the ENHANCED FOCUSED model ***
MODEL_PATH = os.path.join(MODEL_DIR, 'corrected_enhanced_focused_attention_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Model/Data specific config (must match training)
SEQUENCE_LENGTH = 32
NUM_SELECTED_POSE_LANDMARKS = 7 # From focused data collection
# Indices for upper body pose landmarks from MediaPipe Pose (0: Nose, 11-16: Shoulders, Elbows, Wrists)
UPPER_BODY_POSE_LANDMARKS_INDICES = [0, 11, 12, 13, 14, 15, 16]


# Real-time specific config
PREDICTION_THRESHOLD = 0.60 
PREDICTION_BUFFER_SIZE = 10 
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# --- Custom Attention Layer Definition (Must match training script) ---
@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class AttentionWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('A list of two tensors is expected. '
                             f'Got: {inputs}')
        gru_output, attention_probs = inputs
        weighted_sequence = gru_output * attention_probs
        context_vector = tf.reduce_sum(weighted_sequence, axis=1)
        return context_vector

    def get_config(self):
        base_config = super().get_config()
        return base_config

# --- Helper Functions ---

def load_model_with_custom_objects(model_path):
    """Loads a Keras model with custom objects and recompiles."""
    custom_objects = {
        'AttentionWeightedAverage': AttentionWeightedAverage
    }
    try:
        print(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False 
        )
        print("Model loaded successfully.")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Match training LR
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model recompiled.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def extract_focused_keypoints_realtime(results):
    """
    Extracts focused keypoints (Upper Body Pose:7*4, LH:21*3, RH:21*3) = 154 features
    This MUST match the data your model was trained on.
    """
    pose_keypoints_list = []
    if results.pose_landmarks:
        for idx in UPPER_BODY_POSE_LANDMARKS_INDICES:
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                pose_keypoints_list.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                pose_keypoints_list.extend([0,0,0,0]) 
    
    # Ensure consistent length for pose
    if len(pose_keypoints_list) < NUM_SELECTED_POSE_LANDMARKS * 4:
        pose_keypoints_list.extend([0] * (NUM_SELECTED_POSE_LANDMARKS * 4 - len(pose_keypoints_list)))
    
    pose = np.array(pose_keypoints_list).flatten()
    if pose.size == 0: # Fallback
        pose = np.zeros(NUM_SELECTED_POSE_LANDMARKS * 4)


    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh]) # Expected size = (7*4) + (21*3) + (21*3) = 28 + 63 + 63 = 154

def add_temporal_features_realtime(X_scaled):
    """Adds temporal features (Must match training implementation)."""
    # X_scaled has shape (SEQUENCE_LENGTH, num_raw_features=154)
    if X_scaled.shape[0] < 2: 
        X_diff1 = np.zeros_like(X_scaled)
        X_diff2 = np.zeros_like(X_scaled)
    else:
        X_diff1 = np.diff(X_scaled, axis=0, prepend=X_scaled[:1, :])
        X_diff2 = np.diff(X_diff1, axis=0, prepend=X_diff1[:1, :])

    velocity_mag = np.linalg.norm(X_diff1, axis=-1, keepdims=True)
    acceleration_mag = np.linalg.norm(X_diff2, axis=-1, keepdims=True)
    
    # Output shape will be (SEQUENCE_LENGTH, 154*3 + 2 = 464)
    return np.concatenate([X_scaled, X_diff1, X_diff2, velocity_mag, acceleration_mag], axis=-1)

def draw_focused_landmarks_and_prediction(image, results, prediction_text, confidence_value):
    """Draws focused landmarks (upper body pose, hands) and the prediction."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Draw selected Pose connections (can draw all for visual, but only selected are used)
    # To draw only selected upper body pose, you'd need a custom connection list or draw landmarks individually.
    # For simplicity in drawing, we can draw all detected pose landmarks.
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # NO FACE LANDMARKS DRAWN
    
    # Left Hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    # Right Hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    # Prediction Text
    text_to_display = f"Prediction: {prediction_text.upper()} ({confidence_value:.2f})"
    cv2.rectangle(image, (0,0), (image.shape[1], 40), (245, 117, 16), -1) 
    cv2.putText(image, text_to_display, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# --- Main Inference Function ---
def run_inference():
    print(f"Loading resources for ENHANCED FOCUSED model from: {MODEL_DIR}")
    model = load_model_with_custom_objects(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if not all([model, scaler, label_encoder]):
        print("Failed to load one or more resources. Exiting.")
        return

    print("Resources loaded successfully.")
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    print(f"Actions: {actions_map}")

    mp_holistic = mp.solutions.holistic
    # Use lower complexity for Face Mesh if only using hands and pose for landmarks,
    # but Holistic includes it by default. We just won't extract its landmarks.
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        # model_complexity=1 # Can be 0, 1, or 2. Default is 1.
        ) 
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        holistic.close()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    window_name = 'Sign Language Inference - Enhanced Focused'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    sequence_data_raw = deque(maxlen=SEQUENCE_LENGTH) 
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    current_prediction_text = "..."
    current_confidence = 0.0

    print("Starting inference loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # *** Use extract_focused_keypoints_realtime to get 154 features ***
        keypoints = extract_focused_keypoints_realtime(results) 
        sequence_data_raw.append(keypoints)

        if len(sequence_data_raw) == SEQUENCE_LENGTH:
            try:
                X_seq_raw = np.array(sequence_data_raw) # Shape: (SEQUENCE_LENGTH, 154)

                original_shape = X_seq_raw.shape
                X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1]) # Shape: (SEQ_LEN * 1, 154)
                X_scaled = scaler.transform(X_reshaped) # Scaler expects (n_samples, n_features)
                X_scaled = X_scaled.reshape(original_shape) # Shape: (SEQUENCE_LENGTH, 154)

                X_enhanced = add_temporal_features_realtime(X_scaled) # Shape: (SEQUENCE_LENGTH, 464)

                X_input = np.expand_dims(X_enhanced, axis=0) # Shape: (1, SEQUENCE_LENGTH, 464)
                prediction_probabilities = model.predict(X_input)[0]

                predicted_index = np.argmax(prediction_probabilities)
                confidence = prediction_probabilities[predicted_index]

                if confidence > PREDICTION_THRESHOLD:
                    prediction_buffer.append(predicted_index)
                    current_confidence = confidence 
                
                if len(prediction_buffer) > 0:
                    most_common_pred_index = max(set(prediction_buffer), key=prediction_buffer.count)
                    current_prediction_text = actions_map.get(most_common_pred_index, "...")
                    # Update displayed confidence if current frame's high-confidence prediction matches the buffered one
                    if most_common_pred_index == predicted_index and confidence > PREDICTION_THRESHOLD:
                         current_confidence = confidence
                    # If buffered prediction is different, we might show its last known high confidence,
                    # or simply the confidence of the current frame if it's also high.
                    # For now, if it doesn't match, the current_confidence (which might be from a previous frame) is shown.
                else:
                    current_prediction_text = "..."
                    current_confidence = 0.0

            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()
                current_prediction_text = "Error"
                current_confidence = 0.0
        
        display_image = frame.copy() 
        draw_focused_landmarks_and_prediction(display_image, results, current_prediction_text, current_confidence)
        
        cv2.imshow(window_name, display_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Inference finished.")

if __name__ == "__main__":
    run_inference()
