import asyncio
import json
import logging
import uuid
from typing import Dict, List
from fastapi import WebSocket
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class CallManager:
    def __init__(self):
        self.rooms: Dict[str, Dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.user_rooms: Dict[str, str] = {}
        self.room_locks: Dict[str, asyncio.Lock] = {} 
    
    async def _get_room_lock(self, room_id: str) -> asyncio.Lock:
        """Safely gets or creates a lock for a given room_id."""
        if room_id not in self.room_locks:
            self.room_locks[room_id] = asyncio.Lock()
        return self.room_locks[room_id]
        
    async def join_room(self, room_id: str, user_id: str, user_name: str, websocket: WebSocket):
        """Register a user in a room and notify participants.

        We *only* keep the per-room lock while we mutate the in-memory data structures.
        All network I/O (sending confirmation / broadcasting) happens **after** the
        lock is released so one slow websocket cannot block others from joining.
        """

        # Acquire room-specific lock to mutate shared state safely
        lock = await self._get_room_lock(room_id)

        async with lock:
            logger.info(f"User {user_name} ({user_id}) joining room {room_id}")

            # Create the room if it doesn't exist yet
            if room_id not in self.rooms:
                self.rooms[room_id] = {
                    "participants": {},
                    "messages": [],
                    "created_at": asyncio.get_event_loop().time(),
                }
                logger.info(f"Created new room: {room_id}")

            # Register participant in internal maps
            self.rooms[room_id]["participants"][user_id] = {
                "id": user_id,
                "name": user_name,
                "websocket": websocket,
                "sequence_data": deque(maxlen=32),
            }

            self.connections[user_id] = websocket
            self.user_rooms[user_id] = room_id

            # Snapshot of current participants (excluding the new user)
            current_participants = [
                {"id": pid, "name": pdata["name"], "isLocal": False}
                for pid, pdata in self.rooms[room_id]["participants"].items()
                if pid != user_id
            ]

        # ---- lock released ----

        # Inform the newly-joined user about the room state
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "room-joined",
                        "participants": current_participants,
                        "roomId": room_id,
                        "userId": user_id,
                    }
                )
            )
            logger.info(f"Sent room-joined confirmation to {user_name}")
        except Exception as e:
            logger.error(
                f"Failed to send room-joined confirmation to {user_id}: {e}")
            return

        # Notify everyone else that a new participant has joined
        await self.broadcast_to_room(
            room_id,
            {
                "type": "participant-joined",
                "participant": {
                    "id": user_id,
                    "name": user_name,
                    "isLocal": False,
                },
            },
            exclude_user=user_id,
        )

        logger.info(
            f"User {user_name} successfully joined room {room_id}. Total participants: {len(self.rooms[room_id]['participants'])}"
        )
    
    async def leave_room(self, user_id: str):
        if user_id not in self.user_rooms:
            return

        room_id = self.user_rooms[user_id]

        # Acquire lock to mutate the room structures safely
        lock = await self._get_room_lock(room_id)

        async with lock:
            logger.info(f"User {user_id} leaving room {room_id}")

            # Remove participant from the room
            if room_id in self.rooms and user_id in self.rooms[room_id]["participants"]:
                del self.rooms[room_id]["participants"][user_id]

                # Snapshot whether the room will become empty after removal
                room_empty_after_removal = not self.rooms[room_id]["participants"]
            else:
                room_empty_after_removal = False

            # Clean up global tracking regardless of room emptiness
            self.connections.pop(user_id, None)
            self.user_rooms.pop(user_id, None)

        # ---- lock released ----

        # Notify remaining participants (if any) outside the lock
        await self.broadcast_to_room(
            room_id,
            {"type": "participant-left", "participantId": user_id},
        )

        # If the room is now empty, remove it and its lock (no need to hold the lock again)
        if room_id in self.rooms and not self.rooms[room_id]["participants"]:
            self.rooms.pop(room_id, None)
            self.room_locks.pop(room_id, None)
            logger.info(f"Deleted empty room: {room_id}")

        logger.info(f"User {user_id} successfully left room {room_id}")
    
    async def broadcast_to_room(self, room_id: str, message: dict, exclude_user: str = None):
        if room_id not in self.rooms:
            return
            
        disconnected_users = []
        for user_id, participant in list(self.rooms[room_id]["participants"].items()):
            if exclude_user and user_id == exclude_user:
                continue
                
            try:
                await participant["websocket"].send_text(json.dumps(message))
                logger.debug(f"Sent message to {user_id}: {message['type']}")
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.leave_room(user_id)
    
    async def handle_webrtc_message(self, user_id: str, message: dict):
        room_id = self.user_rooms.get(user_id)
        if not room_id:
            logger.warning(f"User {user_id} not in any room for WebRTC message")
            return
            
        target_id = message.get("targetId")
        if target_id and target_id in self.connections:
            message["senderId"] = user_id
            try:
                await self.connections[target_id].send_text(json.dumps(message))
                logger.debug(f"Relayed {message['type']} from {user_id} to {target_id}")
            except Exception as e:
                logger.error(f"Failed to relay WebRTC message from {user_id} to {target_id}: {e}")
        else:
            logger.warning(f"Target {target_id} not found for WebRTC message from {user_id}")
    
    async def handle_prediction_request(self, user_id: str, landmarks: List[float], model, scaler, actions_map, sequence_length: int = 32, confidence_threshold: float = 0.70):
        room_id = self.user_rooms.get(user_id)
        if not room_id or room_id not in self.rooms:
            return
            
        participant = self.rooms[room_id]["participants"].get(user_id)
        if not participant:
            return
            
        # Process prediction using the shared model
        sequence_data = participant["sequence_data"]
        # Offload heavy ML computation to a background thread so we don't block
        # the asyncio event loop handling WebSocket handshakes.
        loop = asyncio.get_running_loop()
        prediction_result = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            self.process_prediction,
            landmarks,
            sequence_data,
            model,
            scaler,
            actions_map,
            sequence_length,
            confidence_threshold,
        )
        
        if prediction_result["status"] == "prediction":
            # Broadcast prediction to all participants in room, including sender.
            # The client will use the senderId to differentiate.
            await self.broadcast_to_room(
                room_id,
                {
                    "type": "prediction",
                    "senderId": user_id,
                    "prediction": prediction_result,
                },
            )
    
    async def handle_chat_message(self, user_id: str, content: str):
        room_id = self.user_rooms.get(user_id)
        if not room_id:
            return
            
        participant = self.rooms[room_id]["participants"].get(user_id)
        if not participant:
            return
            
        message = {
            "type": "chat-message",
            "senderId": user_id,
            "senderName": participant["name"],
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Store message
        self.rooms[room_id]["messages"].append(message)
        
        # Broadcast to room
        await self.broadcast_to_room(room_id, message)
    
    def process_prediction(self, landmarks: List[float], sequence_data: deque, model, scaler, actions_map, sequence_length: int = 32, confidence_threshold: float = 0.70) -> dict:
        """Synchronous helper that runs heavy NumPy / TensorFlow code.

        This function is executed in a worker thread via `run_in_executor` to
        keep the main asyncio loop responsive.
        """
        try:
            # Import here to avoid circular imports
            from predict import add_temporal_features_realtime

            sequence_data.append(landmarks)

            if len(sequence_data) < sequence_length:
                return {
                    "status": "buffering",
                    "progress": len(sequence_data) / sequence_length,
                }

            X_seq_raw = np.array(list(sequence_data))
            X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
            X_scaled = scaler.transform(X_reshaped)
            X_scaled_reshaped = X_scaled.reshape(X_seq_raw.shape)
            X_enhanced = add_temporal_features_realtime(X_scaled_reshaped)
            X_input = np.expand_dims(X_enhanced, axis=0)

            prediction_probs = model.predict_on_batch(X_input)[0]
            predicted_index = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_index]

            if confidence > confidence_threshold:
                return {
                    "status": "prediction",
                    "prediction": actions_map.get(predicted_index, "Unknown"),
                    "confidence": float(confidence * 100),
                }
            else:
                return {"status": "low_confidence"}

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"status": "error", "message": str(e)}

# Global call manager instance
call_manager = CallManager()
