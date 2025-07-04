import asyncio
import json
import logging
import os
import sys
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import joblib
import numpy as np
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import video call functionality
from video_call import call_manager

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR_REL = os.path.join(BASE_DIR, 'model')
sys.path.append(MODEL_DIR_REL)

# Import prediction functions
from predict import load_model_with_custom_objects, add_temporal_features_realtime
from predict_video import predict_on_video

# --- FastAPI App Initialization ---
app = FastAPI(title="Ultra-Fast Sign Language Prediction API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
MODEL_DIR = os.path.join(MODEL_DIR_REL, 'sign_model_focused_enhanced_attention_v2_0.9880_prior1')
MODEL_PATH = os.path.join(MODEL_DIR, 'corrected_enhanced_focused_attention_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SEQUENCE_LENGTH = 32
EXPECTED_LANDMARK_COUNT = 154  # (7*4 for pose) + (21*3 for left hand) + (21*3 for right hand)
CONFIDENCE_THRESHOLD = 0.70

# --- Global State ---
model, scaler, label_encoder, actions_map = None, None, None, {}
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

@app.on_event("startup")
def startup_event():
    global model, scaler, label_encoder, actions_map
    logger.info("Loading prediction model and scaler...")
    try:
        model = load_model_with_custom_objects(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
        logger.info("Resources loaded successfully.")
        logger.info("Video call functionality enabled.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def predict_from_landmarks(landmarks: List[float], sequence_data: deque) -> dict:
    """Receives landmarks, preprocesses, and predicts, providing continuous feedback."""
    sequence_data.append(landmarks)

    if len(sequence_data) < SEQUENCE_LENGTH:
        return {
            "status": "buffering",
            "progress": len(sequence_data) / SEQUENCE_LENGTH
        }

    try:
        X_seq_raw = np.array(list(sequence_data))
        X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_scaled_reshaped = X_scaled.reshape(X_seq_raw.shape)
        X_enhanced = add_temporal_features_realtime(X_scaled_reshaped)
        X_input = np.expand_dims(X_enhanced, axis=0)
        
        prediction_probs = model.predict_on_batch(X_input)[0]
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]

        if confidence > CONFIDENCE_THRESHOLD:
            return {
                "status": "prediction",
                "prediction": actions_map.get(predicted_index, "Unknown"),
                "confidence": float(confidence * 100)
            }
        else:
            return {"status": "low_confidence"}
            
    except Exception as e:
        logger.error(f"Error during prediction pipeline: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# --- WebSocket Endpoints ---

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint for single-user real-time predictions"""
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    loop = asyncio.get_running_loop()

    try:
        while True:
            landmarks_data = await websocket.receive_json()
            if not isinstance(landmarks_data, list) or len(landmarks_data) != EXPECTED_LANDMARK_COUNT:
                logger.warning(f"Received invalid data. Expected list of {EXPECTED_LANDMARK_COUNT}, got {len(landmarks_data) if isinstance(landmarks_data, list) else 'non-list'}")
                continue

            prediction_result = await loop.run_in_executor(
                executor, predict_from_landmarks, landmarks_data, sequence_data
            )
            
            await websocket.send_json(prediction_result)
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)

@app.websocket("/ws/video-call/{room_id}")
async def websocket_video_call(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for video call functionality"""
    user_id = str(uuid.uuid4())
    logger.info(f"Video call WebSocket connection attempt for room {room_id}, user {user_id}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket accepted for user {user_id}")
        
        # Wait for join message with timeout
        try:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            message = json.loads(data)
            logger.info(f"Received initial message from {user_id}: {message}")
            
            if message.get("type") == "join":
                user_name = message.get("userName", f"User_{user_id[:8]}")
                logger.info(f"User {user_name} joining room {room_id}")
                try:
                    await call_manager.join_room(room_id, user_id, user_name, websocket)
                except Exception as e:
                    logger.error(f"Exception in join_room for user {user_id}: {e}", exc_info=True)
                    await websocket.close(code=1011, reason="Internal server error during join_room")
                    return
                
                # Handle subsequent messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        logger.debug(f"Received message from {user_id}: {message.get('type', 'unknown')}")
                        
                        message_type = message.get("type")
                        
                        if message_type in ["offer", "answer", "ice-candidate"]:
                            try:
                                await call_manager.handle_webrtc_message(user_id, message)
                            except Exception as e:
                                logger.error(f"Exception in handle_webrtc_message for user {user_id}: {e}", exc_info=True)
                        elif message_type == "prediction-request":
                            landmarks = message.get("landmarks", [])
                            # The number of landmarks can vary based on MediaPipe config/visibility.
                            # We'll let the processing function handle validation if needed.
                            if landmarks:
                                try:
                                    await call_manager.handle_prediction_request(
                                        user_id, landmarks, model, scaler, actions_map, 
                                        SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD
                                    )
                                except Exception as e:
                                    logger.error(f"Exception in handle_prediction_request for user {user_id}: {e}", exc_info=True)
                            else:
                                logger.warning(f"Empty landmarks received from {user_id}")
                        elif message_type == "chat-message":
                            content = message.get("content", "")
                            if content.strip():
                                try:
                                    await call_manager.handle_chat_message(user_id, content)
                                except Exception as e:
                                    logger.error(f"Exception in handle_chat_message for user {user_id}: {e}", exc_info=True)
                        else:
                            logger.warning(f"Unknown message type from {user_id}: {message_type}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error from {user_id}: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                    except Exception as e:
                        logger.error(f"Error processing message from {user_id}: {e}", exc_info=True)
                        break
            else:
                logger.error(f"Invalid initial message from {user_id}: expected 'join', got {message.get('type')}")
                await websocket.close(code=1003, reason="Invalid initial message")
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for join message from {user_id}")
            await websocket.close(code=1002, reason="Join timeout")
        except Exception as e:
            logger.error(f"Exception while waiting for join message from {user_id}: {e}", exc_info=True)
            await websocket.close(code=1011, reason="Internal server error during join handshake")
            return
        
    except WebSocketDisconnect:
        logger.info(f"Video call WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Video call WebSocket error for user {user_id}: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens
        try:
            await call_manager.leave_room(user_id)
        except Exception as e:
            logger.error(f"Error during cleanup for user {user_id}: {e}", exc_info=True)

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple WebSocket test endpoint"""
    await websocket.accept()
    logger.info("Test WebSocket connection established")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection-test",
            "message": "WebSocket connection successful",
            "timestamp": asyncio.get_event_loop().time()
        }))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Test WebSocket received: {message}")
            
            await websocket.send_text(json.dumps({
                "type": "echo",
                "original": message,
                "timestamp": asyncio.get_event_loop().time()
            }))
            
    except WebSocketDisconnect:
        logger.info("Test WebSocket disconnected")
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")

# --- REST API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "video_calls_enabled": True,
        "active_rooms": len(call_manager.rooms)
    }

@app.get("/api/video-call/rooms")
async def get_active_rooms():
    """Get list of active video call rooms"""
    rooms = []
    for room_id, room_data in call_manager.rooms.items():
        rooms.append({
            "roomId": room_id,
            "participantCount": len(room_data["participants"]),
            "createdAt": room_data["created_at"]
        })
    return {"rooms": rooms}

@app.get("/api/video-call/rooms/{room_id}")
async def check_room_exists(room_id: str):
    """Check if a video call room exists."""
    if room_id in call_manager.rooms:
        return {"exists": True, "participantCount": len(call_manager.rooms[room_id]["participants"])}
    else:
        raise HTTPException(status_code=404, detail="Room not found")

@app.post("/api/video-call/create-room")
async def create_room():
    """Create a new video call room"""
    room_id = str(uuid.uuid4())[:8]
    return {"roomId": room_id}

@app.post("/api/video-predict")
async def video_predict(file: UploadFile = File(...)):
    """Upload and analyze video for sign language predictions"""
    input_path = f"uploads/{file.filename}"
    output_path = f"outputs/annotated_{file.filename}"
    
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process video
        results = predict_on_video(input_path, output_path)
        return JSONResponse({
            "predictions": results,
            "annotated_video_url": f"/api/download/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/api/download/{filename}")
async def download_annotated_video(filename: str):
    """Download annotated video file"""
    file_path = f"outputs/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
