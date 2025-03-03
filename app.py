import argparse
import asyncio
import json
import logging
import uuid
import time
import numpy as np
import cv2
import mediapipe as mp
import weakref

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay


"""

1. Video Processing Service - Handles WebRTC connections and video frame processing
2. Hand Detection Service - Detects hand landmarks using MediaPipe
3. Gesture Recognition Service - Interprets hand landmarks into gestures
4. Cursor Control Service - Translates hand position to cursor coordinates
5. Interaction Feedback Service - Provides visual feedback to the client

"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("touchless-interaction")

# Disable noisy loggers
logging.getLogger('aiortc.rtcrtpreceiver').setLevel(logging.CRITICAL)
logging.getLogger('aioice').setLevel(logging.CRITICAL)

# Global state (would be separate databases/caches in a true microservice architecture)
client_dimensions = {}
active_connections = weakref.WeakSet()


class VideoProcessingService:
    """
    Video Processing Service

    - Capturing and processing webcam feed in real-time
    - Handling WebRTC connections
    - Processing frames from clients
    - Maintaining connection state
    """
    
    def __init__(self):
        """Initialize the Video Processing Service"""
        self.media_relay = MediaRelay()
        self.start_time = time.time()
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "video-processing",
            "status": "online",
            "activeConnections": len(active_connections),
            "uptime": time.time() - self.start_time
        }
    
    def create_peer_connection(self, client_id):
        """Create a new WebRTC peer connection"""
        pc = RTCPeerConnection()
        active_connections.add(pc)
        
        return pc
        
    def subscribe_to_track(self, track):
        """Subscribe to a media track using MediaRelay"""
        return self.media_relay.subscribe(track)
        
    def process_frame(self, frame):
        """Process raw video frame"""
        # Convert YUV to RGB for MediaPipe processing
        if hasattr(frame, 'to_ndarray'):
            # aiortc frame
            img_frame = frame.to_ndarray(format="yuv420p")
            rgb_frame = cv2.cvtColor(img_frame, cv2.COLOR_YUV420P2RGB)
        else:
            # Already ndarray
            rgb_frame = frame
            
        # Flip horizontally for natural interaction
        rgb_frame = cv2.flip(rgb_frame, 1)
        
        return rgb_frame


class HandDetectionService:
    """
    Hand Detection Service

    - Detecting and tracking hand presence in video frames
    - Identifying key hand landmarks
    - Filtering and smoothing detection results
    """
    
    def __init__(self):
        """Initialize the Hand Detection Service with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.start_time = time.time()
        self.detection_count = 0
        self.frame_count = 0
        
    def get_status(self):
        """Get current service status"""
        detection_rate = 0
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            
        return {
            "service": "hand-detection",
            "status": "online",
            "detectionRate": f"{detection_rate:.2f}%",
            "framesProcessed": self.frame_count,
            "uptime": time.time() - self.start_time
        }
        
    def detect_hands(self, frame):
        """Detect hand landmarks in the given frame"""
        self.frame_count += 1
        
        # Process with MediaPipe
        results = self.hands.process(frame)
        
        if results.multi_hand_landmarks:
            self.detection_count += 1
            # Return the first detected hand's landmarks
            return results.multi_hand_landmarks[0].landmark
        
        return None


class GestureRecognitionService:
    """
    Gesture Recognition Service

    - Interpreting hand positions as specific interaction gestures (pinch)
    - Mapping recognized gestures to system commands
    """
    
    def __init__(self):
        """Initialize the Gesture Recognition Service"""
        self.start_time = time.time()
        self.gesture_counts = {
            "pinch": 0,
            "point": 0
        }
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "gesture-recognition",
            "status": "online",
            "supportedGestures": ["pinch", "point"],
            "gesturesCounted": self.gesture_counts,
            "uptime": time.time() - self.start_time
        }
    
    def detect_pinch(self, landmarks):
        """
        Detect pinch gesture between thumb and ring finger
        """
        if not landmarks or len(landmarks) < 21:
            return False
            
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        
        # Calculate distance between thumb and ring finger
        distance = np.sqrt(
            (thumb_tip.x - ring_tip.x)**2 +
            (thumb_tip.y - ring_tip.y)**2
        )
        
        is_pinching = distance < 0.08  # threshold
        
        if is_pinching:
            self.gesture_counts["pinch"] += 1
            
        return is_pinching
        
    def detect_point(self, landmarks):
        """
        Pending
        """
        if not landmarks or len(landmarks) < 21:
            return False
            
        self.gesture_counts["point"] += 1
        return True


class CursorControlService:
    """
    Cursor Control Service

    - Translating hand position to cursor coordinates
    - Applying linear interpolation for natural movement
    - Handling virtual-trackpad boundary conditions
    """
    
    def __init__(self):
        """Initialize the Cursor Control Service"""
        self.start_time = time.time()
        self.cursor_positions = {}  # Last cursor position by client_id
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "cursor-control",
            "status": "online",
            "activeCursors": len(self.cursor_positions),
            "uptime": time.time() - self.start_time
        }
    
    def calculate_cursor_position(self, landmarks, client_id):
        """
        Calculate cursor position based on index finger tip position
        """
        if client_id not in client_dimensions:
            return None, None
        
        dims = client_dimensions[client_id]
        screen_width = dims.get('screenWidth')
        screen_height = dims.get('screenHeight')
        
        # Get virtual box dimensions
        cam_width = dims.get('videoWidth', 1280)
        cam_height = dims.get('videoHeight', 720)
        
        # Make the virtual box larger for easier use
        pad_size = min(cam_width, cam_height) * 0.7
        pad_x = (cam_width - pad_size) / 2
        pad_y = (cam_height - pad_size) / 2
        
        # Get index finger tip position (landmark 8)
        if len(landmarks) <= 8:
            return None, None
            
        index_tip = landmarks[8]
        
        # Transform normalized coordinates to pixel coordinates
        px = int((index_tip.x) * cam_width) 
        py = int(index_tip.y * cam_height)
        
        # Check if finger is in virtual box
        if (pad_x <= px <= pad_x + pad_size and pad_y <= py <= pad_y + pad_size):
            # Map position to screen coordinates
            norm_x = (px - pad_x) / pad_size
            norm_y = (py - pad_y) / pad_size
            
            cursor_x = int(norm_x * screen_width)
            cursor_y = int(norm_y * screen_height)
            
            return self.apply_smoothing(client_id, cursor_x, cursor_y)
        
        return None, None
    
    def apply_smoothing(self, client_id, cursor_x, cursor_y):
        """
        Apply lerp to cursor position
        """
        if cursor_x is None or cursor_y is None:
            return None, None
            
        if client_id in self.cursor_positions:
            last_x, last_y = self.cursor_positions[client_id]
            
            # smoothing with 0.5 factor
            smooth_x = last_x + (cursor_x - last_x) * 0.5
            smooth_y = last_y + (cursor_y - last_y) * 0.5
            
            cursor_x, cursor_y = int(smooth_x), int(smooth_y)
        
        # Store current position for next frame
        self.cursor_positions[client_id] = (cursor_x, cursor_y)
        
        return cursor_x, cursor_y


class InteractionFeedbackService:
    """
    Interaction Feedback Service

    - Providing visual confirmation of detected gestures
    - Guiding users with interface highlights
    - Creating confidence in the touchless interaction
    """
    
    def __init__(self):
        """Initialize the Interaction Feedback Service"""
        self.start_time = time.time()
        self.feedback_count = 0
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "interaction-feedback",
            "status": "online",
            "feedbackCount": self.feedback_count,
            "uptime": time.time() - self.start_time
        }
    
    def prepare_fingertip_data(self, landmarks, frame_shape):
        """
        Prepare fingertip information for client visualization
        Returns list of fingertip coordinates
        """
        if not landmarks:
            return []
            
        # Extract fingertip landmarks
        fingertips = [
            landmarks[4],   # thumb
            landmarks[8],   # index
            landmarks[12],  # middle
            landmarks[16],  # ring
            landmarks[20]   # pinky
        ]
        
        height, width = frame_shape[:2] if len(frame_shape) >= 2 else (720, 1280)
        
        # Convert to client-friendly format
        fingertip_data = []
        for i, tip in enumerate(fingertips):
            fingertip_data.append({
                'x': float(tip.x * width),
                'y': float(tip.y * height),
                'z': float(tip.z)
            })
            
        return fingertip_data
    
    def prepare_feedback_data(self, client_id, landmarks, frame_shape, cursor_x, cursor_y, pinch_status):
        """
        Prepare comprehensive feedback data for client
        Returns dict with cursor position, pinch status, and fingertip data
        """
        self.feedback_count += 1
        
        # Handle case where no hand is detected
        if not landmarks:
            return {
                'handVisible': False,
                'pinchStatus': False
            }
        
        # Prepare feedback with fingertip visualization data
        fingertips = self.prepare_fingertip_data(landmarks, frame_shape)
        
        return {
            'cursorX': cursor_x,
            'cursorY': cursor_y,
            'pinchStatus': pinch_status,
            'handVisible': True,
            'fingertips': fingertips
        }


class SystemIntegrationService:
    """
    System Integration Service

    - Interfacing with existing software
    - Translating gestures into standard input events
    - Coordinating between services
    """
    
    def __init__(self):
        """Initialize system integration and all services"""
        # Initialize all services
        self.video_service = VideoProcessingService()
        self.hand_detection = HandDetectionService()
        self.gesture_recognition = GestureRecognitionService()
        self.cursor_control = CursorControlService()
        self.feedback_service = InteractionFeedbackService()
        
        # For serializing NumPy types
        self.numpy_encoder = NumpyEncoder()
        
    async def process_offer(self, offer_params):
        """Process WebRTC offer and setup connection"""
        offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])
        client_id = offer_params.get("clientId", f"client-{uuid.uuid4()}")
        
        # Store client dimensions if provided
        if "screenWidth" in offer_params and "screenHeight" in offer_params:
            client_dimensions[client_id] = {
                'screenWidth': offer_params.get('screenWidth', 1920),
                'screenHeight': offer_params.get('screenHeight', 1080),
                'videoWidth': offer_params.get('videoWidth', 1280),
                'videoHeight': offer_params.get('videoHeight', 720)
            }
        
        # Create peer connection using Video Processing Service
        pc = self.video_service.create_peer_connection(client_id)
        pc_id = f"PeerConnection({uuid.uuid4()})"
        
        # State for this connection
        data_channel = None
        frame_processing_active = True
        
        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info(f"Connection state changed to {pc.connectionState}")
            nonlocal frame_processing_active
            
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                # Stop processing if connection is closed or failed
                frame_processing_active = False
                
                # Clean up if this connection is in our active set
                if pc in active_connections:
                    active_connections.remove(pc)
                    log_info("Connection removed from active set")
        
        # Handle data channel creation
        @pc.on("datachannel")
        def on_datachannel(channel):
            nonlocal data_channel
            data_channel = channel
            log_info("Data channel opened")
            
            @channel.on("close")
            def on_close():
                nonlocal frame_processing_active
                frame_processing_active = False
                log_info("Data channel closed")
        
        # Handle video track
        @pc.on("track")
        def on_track(track):
            if track.kind != "video":
                return

            log_info(f"Video track added, waiting for media to start flowing")
            
            # Use Video Processing Service to subscribe to track
            receiver = self.video_service.subscribe_to_track(track)
            
            # Store last sent cursor position for smoothing
            last_cursor_x, last_cursor_y = None, None
            
            # Set up task for frame processing
            task = None
            
            async def process_frames():
                nonlocal frame_processing_active, last_cursor_x, last_cursor_y
                frame_count = 0
                
                try:
                    while frame_processing_active:
                        if pc.connectionState in ["failed", "closed"]:
                            log_info("Connection closed, stopping frame processing")
                            break
                            
                        frame_start_time = time.time()
                        
                        try:
                            # Consistent timeout for frame reception
                            frame = await asyncio.wait_for(receiver.recv(), timeout=1.0)
                            
                            # Log first frame received
                            if frame_count == 0:
                                log_info("First frame received - media flow established")
                            
                            # Process frame with Video Processing Service
                            rgb_frame = self.video_service.process_frame(frame)
                            height, width, _ = rgb_frame.shape
                            
                            # Detect hand landmarks with Hand Detection Service
                            landmarks = self.hand_detection.detect_hands(rgb_frame)
                            
                            # Process landmarks if hand is detected
                            if landmarks and data_channel and data_channel.readyState == "open":
                                # Detect gestures with Gesture Recognition Service
                                pinch_status = self.gesture_recognition.detect_pinch(landmarks)
                                
                                # Calculate cursor position with Cursor Control Service
                                cursor_x, cursor_y = self.cursor_control.calculate_cursor_position(
                                    landmarks, client_id
                                )
                                
                                # Prepare feedback with Interaction Feedback Service
                                feedback_data = self.feedback_service.prepare_feedback_data(
                                    client_id,
                                    landmarks,
                                    rgb_frame.shape,
                                    cursor_x,
                                    cursor_y,
                                    pinch_status
                                )
                                
                                # Send cursor data
                                try:
                                    data_channel.send(json.dumps(feedback_data, cls=NumpyEncoder))
                                except Exception as e:
                                    log_info(f"Data send error: {e}")
                                    if "closed" in str(e).lower():
                                        # Channel is closed, stop processing
                                        frame_processing_active = False
                                        break
                            elif data_channel and data_channel.readyState == "open":
                                # Send hand not visible status via Feedback Service
                                try:
                                    data_channel.send(json.dumps({
                                        'handVisible': False,
                                        'pinchStatus': False
                                    }))
                                except Exception as e:
                                    log_info(f"Data send error: {e}")
                                    if "closed" in str(e).lower():
                                        # Channel is closed, stop processing
                                        frame_processing_active = False
                                        break
                            
                            # Log performance metrics occasionally
                            if frame_count % 300 == 0:  # Further reduce log frequency 
                                processing_time = time.time() - frame_start_time
                                log_info(f"Frame processing time - {processing_time:.4f}s")
                            
                            frame_count += 1
                            
                        except asyncio.TimeoutError:
                            # Simplified timeout handling
                            if frame_count == 0:
                                log_info("Waiting for initial video frames...")
                            
                            # Check connection state
                            if pc.connectionState in ["failed", "closed"] or (data_channel and data_channel.readyState != "open"):
                                frame_processing_active = False
                                break
                        except Exception as e:
                            log_info(f"Frame processing error: {str(e)}")
                            # Continue processing other frames instead of breaking
                
                except asyncio.CancelledError:
                    log_info("Frame processing task cancelled")
                finally:
                    log_info("Frame processing stopped")
                    frame_processing_active = False
                    
                    # Clean up if this connection is in our active set
                    if pc in active_connections:
                        active_connections.remove(pc)
            
            # Create and store the task
            task = asyncio.create_task(process_frames())
            
            # Clean up when track ends
            @track.on("ended")
            async def on_ended():
                nonlocal frame_processing_active
                frame_processing_active = False
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Handle WebRTC offer
        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {
                "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type
            }
        except Exception as e:
            logger.exception(f"Error handling offer: {e}")
            raise e
    
    def get_system_status(self):
        """Get comprehensive status of all services"""
        return {
            "system": "touchless-interaction",
            "services": [
                self.video_service.get_status(),
                self.hand_detection.get_status(),
                self.gesture_recognition.get_status(),
                self.cursor_control.get_status(),
                self.feedback_service.get_status()
            ],
            "activeConnections": len(active_connections),
            "registeredClients": len(client_dimensions)
        }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Initialize the main system integration service
system_service = SystemIntegrationService()

# API Routes

async def index(request):
    """Serve the index page"""
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def client_config(request):
    """Handle client configuration updates"""
    params = await request.json()
    client_id = params.get("clientId")
    
    if client_id:
        client_dimensions[client_id] = {
            'screenWidth': params.get('screenWidth', 1920),
            'screenHeight': params.get('screenHeight', 1080),
            'videoWidth': params.get('videoWidth', 1280),
            'videoHeight': params.get('videoHeight', 720)
        }
        return web.Response(text=json.dumps({"status": "success"}))
    
    return web.Response(status=400, text=json.dumps({"status": "error", "message": "Client ID required"}))

async def offer(request):
    """Handle WebRTC offer"""
    params = await request.json()
    
    try:
        result = await system_service.process_offer(params)
        return web.Response(
            content_type="application/json",
            text=json.dumps(result)
        )
    except Exception as e:
        logger.exception(f"Error handling offer: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(e)})
        )

async def status(request):
    """Return system status"""
    return web.Response(
        content_type="application/json",
        text=json.dumps(system_service.get_system_status(), cls=NumpyEncoder)
    )

async def cleanup_inactive_connections():
    """Periodically check and clean up inactive connections"""
    while True:
        try:
            # Log current connection count every minute
            if len(active_connections) > 0:
                logger.info(f"Active connections: {len(active_connections)}")
            
            # Check each connection
            to_remove = []
            for pc in active_connections:
                if pc.connectionState in ["failed", "closed"]:
                    to_remove.append(pc)
            
            # Remove dead connections
            for pc in to_remove:
                active_connections.remove(pc)
                
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Continue even if there's an error

async def start_background_tasks(app):
    """Start background tasks when the app starts"""
    app['cleanup_task'] = asyncio.create_task(cleanup_inactive_connections())
    logger.info("Started background cleanup task")

async def cleanup_background_tasks(app):
    """Clean up background tasks when the app stops"""
    app['cleanup_task'].cancel()
    try:
        await app['cleanup_task']
    except asyncio.CancelledError:
        pass
    logger.info("Canceled background cleanup task")
    
    # Close all active connections
    for pc in list(active_connections):
        pc.close()
    logger.info("Closed all active connections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Touchless Interaction System")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    args = parser.parse_args()

    # Initialize the application
    app = web.Application()
    
    # Setup routes
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/config", client_config)
    app.router.add_get("/status", status)

    # Enable background tasks
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    # Initialize MediaPipe with a test frame to ensure it's loaded
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        system_service.hand_detection.detect_hands(dummy_frame)
    logger.info("MediaPipe hands initialized")
    
    # Start the server
    web.run_app(
        app, 
        host=args.host, 
        port=args.port
    )