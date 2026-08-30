import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.face_database import known_faces, known_names

# ==========================================
# Initialize InsightFace
# ==========================================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

# ==========================================
# Recognition Threshold
# ==========================================

SIMILARITY_THRESHOLD = 0.55

# ==========================================
# Recognize Face INSIDE Person Crop
# ==========================================

def recognize_faces(person_crop):

    if person_crop is None:
        return []

    if person_crop.size == 0:
        return []

# ==========================================
# Enlarge person crop before face detection
# ==========================================

    person_crop = cv2.resize(
        person_crop,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    faces = face_app.get(person_crop)

    results = []

    for face in faces:

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        best_score = -1
        best_name = "Unknown"

        for known_embedding, known_name in zip(
            known_faces,
            known_names
        ):

            score = np.dot(
                embedding,
                known_embedding
            )

            if score > best_score:
                best_score = score
                best_name = known_name

        if best_score < SIMILARITY_THRESHOLD:
            best_name = "Unknown"

        x1, y1, x2, y2 = face.bbox.astype(int)

        # Convert coordinates back
        x1 //= 2
        y1 //= 2
        x2 //= 2
        y2 //= 2

        results.append({

            "name": best_name.replace("_", " ").title(),

            "confidence": round(float(best_score), 2),

            "bbox": (
                x1,
                y1,
                x2,
                y2
            )

        })

    return results