import pandas as pd

# Load original dataset
df = pd.read_csv("hand_landmarks.csv", header=None)

# Normalize landmarks relative to wrist
def normalize_row(row):
    label = row[0]
    coords = row[1:].values.reshape(-1, 3).astype(float)

    # Wrist as origin
    base = coords[0]
    coords = coords - base

    # Scale by distance between wrist (0) and middle finger tip (12)
    scale = ((coords[12]**2).sum())**0.5
    if scale > 0:
        coords /= scale

    flat = coords.flatten()
    return [label] + flat.tolist()

# Apply normalization
processed = df.apply(normalize_row, axis=1, result_type="expand")

# Save normalized dataset
processed.to_csv("hand_landmarks_normalized.csv", index=False, header=False)
print("✅ Normalized dataset saved as hand_landmarks_normalized.csv")
