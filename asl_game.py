def extract_landmarks(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    zs = [lm.z for lm in hand_landmarks.landmark]

    coords = list(zip(xs, ys, zs))

    # Wrist as origin
    base_x, base_y, base_z = coords[0]
    coords = [(x-base_x, y-base_y, z-base_z) for (x,y,z) in coords]

    # Scale by distance wrist→middle finger tip
    scale = (coords[12][0]**2 + coords[12][1]**2 + coords[12][2]**2)**0.5
    if scale > 0:
        coords = [(x/scale, y/scale, z/scale) for (x,y,z) in coords]

    # Flatten
    landmarks = [v for triplet in coords for v in triplet]
    return landmarks
