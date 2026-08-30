from app.face_database import known_faces, known_names

print()

print("Names Loaded:")
print("----------------")

for name in known_names:
    print(name)

print()
print(f"Total Faces: {len(known_faces)}")