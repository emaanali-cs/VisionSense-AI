import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ==========================
# Initialize InsightFace
# ==========================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

# ==========================
# Dataset Location
# ==========================

DATASET_PATH = os.path.join("assets", "staff_faces")

# ==========================
# Load Face Database
# ==========================

known_faces = []
known_names = []

print("=" * 50)
print("Loading Face Database...")
print("=" * 50)

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Loading: {person_name}")

    person_embeddings = []

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = face_app.get(image)

        if len(faces) == 0:
            print(f"   No face found -> {image_name}")
            continue

        embedding = faces[0].embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        person_embeddings.append(embedding)

    # Average all embeddings of this person
    if len(person_embeddings) > 0:

        avg_embedding = np.mean(person_embeddings, axis=0)

        # Normalize again
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        known_faces.append(avg_embedding)
        known_names.append(person_name)

print("=" * 50)
print(f"Loaded {len(known_faces)} people.")
print("=" * 50)