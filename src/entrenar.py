"""
entrenar.py — Clasificador de 33 frutas con Data Augmentation
Requiere: pip install tensorflow numpy pillow scipy
Ejecutar desde: src/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuración ───────────────────────────────────────────────────────────
IMG_SIZE   = (100, 100)
BATCH_SIZE = 32
EPOCHS     = 30
TRAIN_DIR  = "../dataset/training"
TEST_DIR   = "../dataset/test"
MODEL_OUT  = "../modelos/clasificador_v1.h5"

print("=" * 55)
print("  ENTRENAMIENTO 33 FRUTAS + DATA AUGMENTATION")
print("=" * 55)

# ── 1. Generadores ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    shear_range=0.15,
    brightness_range=[0.4, 1.6],   # simula distintas iluminaciones
    channel_shift_range=40.0,       # simula distintos fondos/colores
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True,
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False,
)

NUM_CLASES = len(train_gen.class_indices)
print(f"\n✅ Clases encontradas: {NUM_CLASES}")
print(f"   Imágenes de entrenamiento: {train_gen.samples}")
print(f"   Imágenes de test:          {test_gen.samples}")

# ── 2. Modelo CNN ───────────────────────────────────────────────────────────
model = models.Sequential([

    # Bloque 1
    layers.Conv2D(32, (3,3), activation="relu", padding="same",
                  input_shape=(100, 100, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Bloque 2
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Bloque 3
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    # Bloque 4
    layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    # Clasificador
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASES, activation="softmax"),
])

model.summary()

# ── 3. Compilar ─────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 4. Callbacks ────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
]

# ── 5. Entrenar ─────────────────────────────────────────────────────────────
print("\n🚀 Iniciando entrenamiento...\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1,
)

# ── 6. Evaluar ──────────────────────────────────────────────────────────────
print("\n📊 Evaluando en test set...")
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"   Accuracy final: {acc*100:.2f}%")
print(f"   Loss final:     {loss:.4f}")

# ── 7. Guardar ──────────────────────────────────────────────────────────────
os.makedirs("../modelos", exist_ok=True)
model.save(MODEL_OUT)
print(f"\n✅ Modelo guardado en '{MODEL_OUT}'")
print(f"\n⚠️  Orden de clases para app.py:")
for nombre, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
    print(f"   {idx} → {nombre}")