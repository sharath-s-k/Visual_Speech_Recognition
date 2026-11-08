import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten
)

# =========================
# 1. Vocabulary
# =========================

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=vocab,
    oov_token=""
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True
)


# =========================
# 2. Load video
# =========================

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame is None:
            continue

        try:
            if len(frame.shape) != 3:
                continue

            if frame.shape[0] < 236 or frame.shape[1] < 220:
                continue

            if frame.shape[2] != 3:
                continue

            frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
            frame = tf.image.rgb_to_grayscale(frame)

            frame = frame[190:236, 80:220, :]

            if tuple(frame.shape) != (46, 140, 1):
                continue

            frames.append(frame)

        except Exception:
            continue

    cap.release()

    if not frames:
        raise ValueError(f"No valid frames found in {path}")

    frames = tf.stack(frames)
    frames = tf.cast(frames, tf.float32)

    mean = tf.reduce_mean(frames)
    std = tf.math.reduce_std(frames)

    return (frames - mean) / (std + 1e-8)


# =========================
# 3. Model
# =========================

model = Sequential()

model.add(
    Conv3D(
        128,
        3,
        input_shape=(75, 46, 140, 1),
        padding="same"
    )
)
model.add(Activation("relu"))
model.add(MaxPool3D(pool_size=(1, 2, 2)))

model.add(Conv3D(256, 3, padding="same"))
model.add(Activation("relu"))
model.add(MaxPool3D(pool_size=(1, 2, 2)))

model.add(Conv3D(75, 3, padding="same"))
model.add(Activation("relu"))
model.add(MaxPool3D(pool_size=(1, 2, 2)))

model.add(TimeDistributed(Flatten()))

model.add(
    Bidirectional(
        LSTM(
            128,
            kernel_initializer="orthogonal",
            return_sequences=True
        )
    )
)
model.add(Dropout(0.5))

model.add(
    Bidirectional(
        LSTM(
            128,
            kernel_initializer="orthogonal",
            return_sequences=True
        )
    )
)
model.add(Dropout(0.5))

model.add(
    Dense(
        char_to_num.vocabulary_size() + 1,
        kernel_initializer="he_normal",
        activation="softmax"
    )
)


# =========================
# 4. Load trained weights
# =========================

checkpoint_path = "models/checkpoint.weights.h5"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

model.load_weights(checkpoint_path)

print("\nTrained model loaded successfully.")


# =========================
# 5. Select video
# =========================

video_path = input(
    "\nEnter video path "
    "(example: data/s1/bbaf2n.mpg): "
).strip()

if not os.path.exists(video_path):
    raise FileNotFoundError(
        f"Video not found: {video_path}"
    )


# =========================
# 6. Process video
# =========================

frames = load_video(video_path)

print("Frames loaded:", len(frames))

# Model expects maximum 75 frames
frames = frames[:75]

# If fewer than 75 frames, pad
if len(frames) < 75:
    padding = tf.zeros(
        (75 - len(frames), 46, 140, 1),
        dtype=tf.float32
    )
    frames = tf.concat([frames, padding], axis=0)

frames = tf.expand_dims(frames, axis=0)


# =========================
# 7. Prediction
# =========================

yhat = model.predict(frames, verbose=0)

decoded = tf.keras.backend.ctc_decode(
    yhat,
    input_length=[75],
    greedy=True
)[0][0].numpy()


# =========================
# 8. Convert prediction
# =========================

prediction = tf.strings.reduce_join(
    [num_to_char(word) for word in decoded[0]]
).numpy().decode("utf-8")

print("\n" + "=" * 60)
print("LIP READING RESULT")
print("=" * 60)
print("Video     :", video_path)
print("Prediction:", prediction)
print("=" * 60)