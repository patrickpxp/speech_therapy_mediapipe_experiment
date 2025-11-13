<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# i want to create an app to help kids with speech issues , kinda like speech therapy. for this MVP, we should focus on identifying face and mouth region landmarks using computer vision . that the students is doing the right thing , for example to spell the letter A or O. how would i proceed

Perfect! Now I have all the information needed. Let me create a comprehensive guide for building the speech therapy app.

## Building a Speech Therapy App with Computer Vision: Technical Implementation Guide

Creating a speech therapy MVP focused on mouth landmark detection and phoneme validation is an excellent project. Here's a comprehensive roadmap for your implementation, tailored to your technical background as a developer and quantitative professional.

### Architecture Overview

Your MVP should follow a **modular three-layer architecture**:

1. **Frontend**: Real-time video capture with visual feedback
2. **Backend**: MediaPipe-based landmark detection + phoneme validation
3. **Analytics**: Performance tracking and user progress

For your tech stack, leverage **FastAPI** for the backend (aligns with your development preferences), **MediaPipe** for computer vision, and **React** or **Streamlit** for rapid frontend prototyping.

### Phase 1: Mouth Landmark Detection Infrastructure

**Primary Tool: MediaPipe Face Mesh**[^1_1][^1_2][^1_3][^1_4]

MediaPipe provides **478 3D facial landmarks** in real-time with excellent accuracy. For your use case, focus on the **mouth region landmarks**:[^1_5][^1_6]

**Mouth Landmark Indices**:[^1_7][^1_8]

- **Outer Lips**: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
- **Inner Lips**: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
- **Upper Lip Outer**:[^1_9]
- **Lower Lip Outer**:[^1_10]

```python
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmark indices
LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER_OUTER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

def extract_mouth_landmarks(frame):
    h, w, c = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[^1_0]
        mouth_points = {
            'upper_outer': [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                           for i in LIPS_UPPER_OUTER],
            'lower_outer': [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                           for i in LIPS_LOWER_OUTER],
            'upper_inner': [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                           for i in LIPS_UPPER_INNER],
            'lower_inner': [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                           for i in LIPS_LOWER_INNER]
        }
        return mouth_points
    return None
```


### Phase 2: Viseme Recognition and Phoneme Classification

**Core Concept: Visemes vs. Phonemes**[^1_11][^1_12][^1_13]

A **viseme** is the visual representation of a phoneme—how the mouth looks when producing a specific sound. Critical insight: **multiple phonemes may map to one viseme**. For example, "f" and "v" look nearly identical mouth-wise.[^1_12][^1_11]

**Key Mouth Shape Features to Extract**:[^1_14]

- **Lip Opening Distance**: Euclidean distance between upper and lower lip centers
- **Lip Width**: Horizontal span of lips
- **Mouth Aspect Ratio**: Height/width relationship
- **Lip Curvature**: Shape classification (open, rounded, spread, etc.)
- **Mouth Corner Position**: Elevation/depression of mouth corners

```python
def extract_mouth_features(mouth_landmarks):
    """Extract geometric features from mouth landmarks"""
    
    if not mouth_landmarks:
        return None
    
    upper = np.array(mouth_landmarks['upper_outer'])
    lower = np.array(mouth_landmarks['lower_outer'])
    
    # Lip opening distance (vertical)
    lip_opening = np.mean([
        np.linalg.norm(upper[i] - lower[i]) 
        for i in range(len(upper))
    ])
    
    # Lip width (horizontal)
    lip_width = np.linalg.norm(upper[^1_0] - upper[-1])
    
    # Mouth aspect ratio
    aspect_ratio = lip_opening / (lip_width + 1e-6)
    
    # Lip curvature (using polynomial fit)
    upper_curve = np.polyfit(upper[:, 0], upper[:, 1], 2)[^1_0]
    lower_curve = np.polyfit(lower[:, 0], lower[:, 1], 2)[^1_0]
    
    return {
        'lip_opening': lip_opening,
        'lip_width': lip_width,
        'aspect_ratio': aspect_ratio,
        'upper_curvature': upper_curve,
        'lower_curvature': lower_curve
    }
```


### Phase 3: Phoneme Validation Engine

**Phoneme-to-Viseme Mapping**[^1_13][^1_11][^1_12]

Create a database mapping key vowels and consonants to expected mouth shapes:

```python
PHONEME_PATTERNS = {
    'A': {
        'target_aspect_ratio': (0.35, 0.50),
        'target_lip_width': (80, 120),
        'mouth_shape': 'open_jaw',
        'description': 'Wide open mouth, jaw dropped'
    },
    'O': {
        'target_aspect_ratio': (0.20, 0.35),
        'target_lip_width': (50, 90),
        'mouth_shape': 'rounded',
        'description': 'Rounded lips, mouth moderately open'
    },
    'E': {
        'target_aspect_ratio': (0.10, 0.25),
        'target_lip_width': (90, 130),
        'mouth_shape': 'spread',
        'description': 'Stretched horizontally, minimal opening'
    },
    'M': {
        'target_aspect_ratio': (0.0, 0.05),
        'target_lip_width': (70, 110),
        'mouth_shape': 'closed',
        'description': 'Lips closed, bilabial contact'
    },
    'S': {
        'target_aspect_ratio': (0.05, 0.15),
        'target_lip_width': (80, 130),
        'mouth_shape': 'slightly_open',
        'description': 'Slight opening, teeth showing'
    }
}

def validate_phoneme(features, phoneme, tolerance=0.15):
    """Validate if current mouth position matches target phoneme"""
    
    if phoneme not in PHONEME_PATTERNS:
        return {'valid': False, 'error': 'Unknown phoneme'}
    
    pattern = PHONEME_PATTERNS[phoneme]
    aspect_ratio = features['aspect_ratio']
    lip_width = features['lip_width']
    
    # Check aspect ratio
    min_ar, max_ar = pattern['target_aspect_ratio']
    ar_valid = min_ar <= aspect_ratio <= max_ar
    ar_distance = min(
        abs(aspect_ratio - min_ar),
        abs(aspect_ratio - max_ar)
    ) if not ar_valid else 0
    
    # Check lip width
    min_w, max_w = pattern['target_lip_width']
    width_valid = min_w <= lip_width <= max_w
    width_distance = min(
        abs(lip_width - min_w),
        abs(lip_width - max_w)
    ) if not width_valid else 0
    
    # Calculate confidence score (0-100)
    max_error = max(max_ar - min_ar, max_w - min_w)
    error = (ar_distance + width_distance) / (2 * max_error + 1e-6)
    confidence = max(0, 100 * (1 - error / tolerance))
    
    return {
        'valid': ar_valid and width_valid,
        'confidence': confidence,
        'pattern': pattern,
        'feedback': generate_feedback(features, pattern)
    }

def generate_feedback(features, pattern):
    """Generate real-time feedback for user correction"""
    aspect_ratio = features['aspect_ratio']
    target_ar = sum(pattern['target_aspect_ratio']) / 2
    
    feedback = []
    if aspect_ratio < pattern['target_aspect_ratio'][^1_0]:
        feedback.append("Open your mouth more")
    elif aspect_ratio > pattern['target_aspect_ratio'][^1_1]:
        feedback.append("Close your mouth a bit")
    
    lip_width = features['lip_width']
    target_width = sum(pattern['target_lip_width']) / 2
    if lip_width < pattern['target_lip_width'][^1_0]:
        feedback.append("Spread your lips wider")
    elif lip_width > pattern['target_lip_width'][^1_1]:
        feedback.append("Narrow your lips")
    
    return feedback if feedback else ["Good position!"]
```


### Phase 4: FastAPI Backend Implementation

Structure your backend with clear separation of concerns:

```python
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import base64
from datetime import datetime

app = FastAPI(title="Speech Therapy API")

class SpeechTherapySession:
    def __init__(self, user_id: str, target_phonemes: list):
        self.user_id = user_id
        self.target_phonemes = target_phonemes
        self.session_data = []
        self.current_phoneme_index = 0
        self.start_time = datetime.now()
    
    def add_result(self, phoneme, features, validation):
        self.session_data.append({
            'timestamp': datetime.now(),
            'phoneme': phoneme,
            'features': features,
            'validation': validation
        })

# Store active sessions
active_sessions = {}

@app.post("/api/v1/session/start")
async def start_session(user_id: str, phonemes: list):
    """Initialize a new therapy session"""
    session = SpeechTherapySession(user_id, phonemes)
    active_sessions[user_id] = session
    return {
        'session_id': user_id,
        'target_phonemes': phonemes,
        'current_target': phonemes[^1_0] if phonemes else None
    }

@app.websocket("/ws/therapy/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Real-time video stream processing via WebSocket"""
    await websocket.accept()
    session = active_sessions.get(user_id)
    
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    try:
        while True:
            # Receive base64 encoded frame
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process landmarks
            mouth_landmarks = extract_mouth_landmarks(frame)
            if mouth_landmarks:
                features = extract_mouth_features(mouth_landmarks)
                current_phoneme = session.target_phonemes[session.current_phoneme_index]
                validation = validate_phoneme(features, current_phoneme)
                
                session.add_result(current_phoneme, features, validation)
                
                # Send feedback to client
                response = {
                    'phoneme': current_phoneme,
                    'confidence': validation['confidence'],
                    'valid': validation['valid'],
                    'feedback': validation['feedback'],
                    'features': {
                        'lip_opening': features['lip_opening'],
                        'aspect_ratio': features['aspect_ratio']
                    }
                }
                await websocket.send_json(response)
    
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))

@app.post("/api/v1/session/{user_id}/complete")
async def complete_session(user_id: str):
    """End session and return analytics"""
    session = active_sessions.get(user_id)
    if not session:
        return {'error': 'Session not found'}
    
    # Calculate metrics
    total_frames = len(session.session_data)
    successful_frames = sum(1 for d in session.session_data if d['validation']['valid'])
    avg_confidence = np.mean([d['validation']['confidence'] 
                             for d in session.session_data])
    
    analytics = {
        'session_duration': (datetime.now() - session.start_time).total_seconds(),
        'total_frames_processed': total_frames,
        'successful_phoneme_attempts': successful_frames,
        'success_rate': (successful_frames / total_frames * 100) if total_frames > 0 else 0,
        'average_confidence': avg_confidence,
        'phonemes_practiced': session.target_phonemes
    }
    
    del active_sessions[user_id]
    return analytics
```


### Phase 5: Frontend Real-Time Display

For rapid MVP development, use **Streamlit** for instant results:

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Speech Therapy - Phoneme Practice")

# Session initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())

target_phoneme = st.selectbox("Choose a phoneme to practice", ['A', 'O', 'E', 'M', 'S'])

# Real-time camera feed
camera_input = st.camera_input("Position your face in frame")

if camera_input is not None:
    # Process frame
    image = Image.open(camera_input)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    mouth_landmarks = extract_mouth_landmarks(frame)
    if mouth_landmarks:
        features = extract_mouth_features(mouth_landmarks)
        validation = validate_phoneme(features, target_phoneme)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image)
        
        with col2:
            st.metric("Confidence", f"{validation['confidence']:.1f}%")
            st.metric("Status", "✅ Correct!" if validation['valid'] else "❌ Adjust")
            
            for feedback in validation['feedback']:
                st.info(feedback)
```


### Critical Implementation Considerations

**1. Robustness to Head Pose**[^1_15][^1_16]

MediaPipe handles moderate head tilting well, but extreme angles reduce accuracy. Implement head pose estimation:

```python
def estimate_head_pose(landmarks):
    """Simple head pose validation"""
    left_cheek = np.array([landmarks.landmark[^1_234].x, landmarks.landmark[^1_234].y])
    right_cheek = np.array([landmarks.landmark[^1_454].x, landmarks.landmark[^1_454].y])
    cheek_distance = np.linalg.norm(left_cheek - right_cheek)
    
    # If cheeks are too asymmetrical, head is tilted
    return cheek_distance > 0.15  # Returns True if frontal
```

**2. Temporal Smoothing**[^1_2][^1_17]

Reduce jitter by averaging landmarks across frames:

```python
from collections import deque

class LandmarkBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, landmarks):
        self.buffer.append(landmarks)
    
    def get_smoothed(self):
        if not self.buffer:
            return None
        return np.mean(list(self.buffer), axis=0)
```

**3. Performance Optimization**

- MediaPipe processes at ~30 FPS on CPU; GPU acceleration can reach 60+ FPS
- Deploy on **GCP Cloud Run** or **Hetzner** for scalability
- Use Docker for consistent environments


### Deployment Architecture

**Docker Configuration** (aligns with your deployment experience):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt**:

```
fastapi==0.104.1
uvicorn==0.24.0
opencv-python==4.8.1
mediapipe==0.10.5
numpy==1.24.3
pydantic==2.5.0
```


### Next Steps for Production

1. **Expand Phoneme Database**: Add 20+ phonemes with teacher-created reference videos
2. **User Progress Tracking**: Store session history in PostgreSQL with analytics dashboard
3. **Gamification**: Add badges, streaks, and leaderboards (especially effective for kids)
4. **Audio Integration**: Combine mouth validation with audio phoneme recognition using **Whisper API**
5. **Mobile Deployment**: Use **React Native** or **Flutter** with WebRTC for real device deployment

This architecture gives you a solid MVP foundation that directly validates mouth positioning against target phoneme shapes while providing real-time feedback to students.[^1_3][^1_18][^1_19][^1_16][^1_1][^1_2][^1_14]
<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

[^1_2]: https://www.geeksforgeeks.org/machine-learning/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/

[^1_3]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

[^1_4]: https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html

[^1_5]: https://github.com/google/mediapipe/issues/2040

[^1_6]: https://github.com/google-ai-edge/mediapipe/issues/963

[^1_7]: https://www.irjmets.com/upload_newfiles/irjmets70700024025/paper_file/irjmets70700024025.pdf

[^1_8]: https://stackoverflow.com/questions/66649492/how-to-get-specific-landmark-of-face-like-lips-or-eyes-using-tensorflow-js-face

[^1_9]: https://blog.roboflow.com/what-is-mediapipe/

[^1_10]: https://apps.apple.com/us/app/otsimo-speech-therapy-slp/id1440975537

[^1_11]: https://download.autodesk.com/global/docs/softimage2013/en_us/userguide/files/face_lipsync_AboutPhonemesandVisemes.htm

[^1_12]: https://patents.google.com/patent/US20210390949A1/en

[^1_13]: https://www.sciencedirect.com/science/article/abs/pii/S0167639317300286

[^1_14]: https://www.iieta.org/journals/ts/paper/10.18280/ts.380411

[^1_15]: https://arxiv.org/abs/1911.06095

[^1_16]: https://arxiv.org/html/2507.18863v1

[^1_17]: https://learnopencv.com/introduction-to-mediapipe/

[^1_18]: https://www.granitebayspeech.com/therapy-apps/

[^1_19]: https://play.google.com/store/apps/details?id=com.otsimo.speech\&hl=en

[^1_20]: https://arxiv.org/html/2501.11899v1

[^1_21]: https://stackoverflow.com/questions/74901522/can-mediapipe-specify-which-parts-of-the-face-mesh-are-the-lips-or-nose-or-eyes

[^1_22]: https://dl.acm.org/doi/fullHtml/10.1145/3613904.3642092

[^1_23]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

[^1_24]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4474762/

[^1_25]: https://github.com/nikitansg/Face-detection-mediapipe

[^1_26]: https://www.youtube.com/watch?v=rAS17tDYeA0

[^1_27]: https://avsp2017.loria.fr/wp-content/uploads/2017/07/AVSP2017_paper_17.pdf

[^1_28]: https://dergipark.org.tr/tr/download/article-file/2904250

[^1_29]: https://play.google.com/store/apps/details?id=org.blubblub.app.speechblubs\&hl=en

[^1_30]: https://www.simiokunowo.me/blog/building-a-speaker-diarization-api

[^1_31]: https://github.com/remsky/Kokoro-FastAPI

[^1_32]: https://www.ijeit.com/Vol 3/Issue 1/IJEIT1412201307_87.pdf

[^1_33]: https://python.plainenglish.io/building-a-real-time-chat-app-backend-with-python-and-fastapi-83af81ecec70

[^1_34]: https://www.cs.mcgill.ca/~jvybihal/assets/papers/IEEE_TMM.pdf

[^1_35]: https://learn.microsoft.com/en-us/answers/questions/2286520/best-way-to-deploy-a-python-fastapi-application-wi

[^1_36]: https://quickpose.ai/faqs/mediapipe-vs-opencv/

[^1_37]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8034549/

[^1_38]: https://www.isca-archive.org/speechprosody_2016/yang16_speechprosody.pdf

