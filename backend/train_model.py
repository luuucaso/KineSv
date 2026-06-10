import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"

X = []
y = []

MAX_FRAMES = 30

# Leer datasets

for filename in os.listdir(DATASET_PATH):

    if filename.endswith(".json"):

        path = os.path.join(DATASET_PATH, filename)

        with open(path, "r") as f:

            data = json.load(f)

        frames = data["frames"]
        label = data["label"]

        # Asegurar 30 frames

        if len(frames) < MAX_FRAMES:
            continue

        frames = frames[:MAX_FRAMES]

        X.append(frames)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Shape:", X.shape)
print("Primer frame:", len(X[0][0]))

# Convertir labels

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2
)

# Modelo LSTM

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        activation="relu",
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(
    LSTM(
        128,
        return_sequences=False,
        activation="relu"
    )
)

model.add(Dense(64, activation="relu"))

model.add(
    Dense(
        y_categorical.shape[1],
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entrenar

model.fit(
    X_train,
    y_train,
    epochs=50
)

# Evaluar

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

# Guardar modelo

model.save("model/modelo_lsu.h5")

# Guardar labels

with open("model/labels.json", "w") as f:

    json.dump(
        encoder.classes_.tolist(),
        f
    )

print("Modelo guardado")