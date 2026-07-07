# ============================================================
# DAY 17 - IMAGE AUGMENTATION PIPELINE
# Part 1 : Data Loading + Augmentation + Visualization
# ============================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2
import random

# ============================================================
# Set Random Seed
# ============================================================

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Dataset Parameters
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "Dataset"

# ============================================================
# Load Dataset
# ============================================================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ============================================================
# Class Names
# ============================================================

class_names = train_dataset.class_names

print("\n===================================")
print("Classes Found")
print("===================================")

for i, cls in enumerate(class_names):
    print(f"{i+1}. {cls}")

print("\nTotal Classes :", len(class_names))

# ============================================================
# Optimize Dataset Performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ============================================================
# TensorFlow Augmentation Pipeline
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.20),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomContrast(0.20)
])

print("\nTensorFlow Augmentation Pipeline Created.")

# ============================================================
# Albumentations Pipeline
# ============================================================

albumentation_transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.10,
        scale_limit=0.10,
        rotate_limit=20,
        p=1.0
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=1.0
    )
])

print("Albumentations Pipeline Created.")

# ============================================================
# Visualize 16 TensorFlow Augmented Images
# ============================================================

print("\nGenerating Augmented Images...")

images, labels = next(iter(train_dataset))

sample_image = images[0]

plt.figure(figsize=(10,10))

for i in range(16):

    augmented = data_augmentation(
        tf.expand_dims(sample_image, axis=0),
        training=True
    )

    plt.subplot(4,4,i+1)
    plt.imshow(augmented[0].numpy().astype("uint8"))
    plt.axis("off")

plt.suptitle("TensorFlow Augmented Images", fontsize=16)

plt.tight_layout()

plt.savefig("augmented_images.png")

plt.show()

print("\n16 Augmented Images Saved Successfully.")

# ============================================================
# Visualize Albumentations Example
# ============================================================

image_np = sample_image.numpy().astype(np.uint8)

augmented = albumentation_transform(image=image_np)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image_np)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(augmented["image"])
plt.title("Albumentations")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nPart 1 Completed Successfully.")

# ============================================================
# Part 2 : Build CNN & Train Models
# ============================================================

# ============================================================
# Normalize Dataset
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds_normal = train_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds_normal = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ============================================================
# Dataset with TensorFlow Augmentation
# ============================================================

train_ds_augmented = train_dataset.map(
    lambda x, y: (
        normalization_layer(
            data_augmentation(x, training=True)
        ),
        y
    ),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ============================================================
# CNN Model Function
# ============================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(224,224,3)),

        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256,activation='relu'),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(len(class_names),activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================
# Train WITHOUT Augmentation
# ============================================================

print("\n==========================================")
print("Training WITHOUT Augmentation")
print("==========================================")

baseline_model = build_model()

history_without = baseline_model.fit(

    train_ds_normal,

    validation_data=val_ds_normal,

    epochs=10,

    verbose=1

)

# ============================================================
# Train WITH Augmentation
# ============================================================

print("\n==========================================")
print("Training WITH Augmentation")
print("==========================================")

augmented_model = build_model()

history_with = augmented_model.fit(

    train_ds_augmented,

    validation_data=val_ds_normal,

    epochs=10,

    verbose=1

)

print("\nPart 2 Completed Successfully.")


# ============================================================
# Part 3 : Results, Comparison & Save Model
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Best Validation Accuracy
# ============================================================

best_without = max(history_without.history['val_accuracy'])
best_with = max(history_with.history['val_accuracy'])

print("\n========================================")
print("Validation Accuracy Comparison")
print("========================================")
print(f"Without Augmentation : {best_without:.4f}")
print(f"With Augmentation    : {best_with:.4f}")

# ============================================================
# Percentage Gain
# ============================================================

gain = ((best_with - best_without) / best_without) * 100

print(f"\nPercentage Gain : {gain:.2f}%")

# ============================================================
# Accuracy Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    history_without.history['val_accuracy'],
    label='Without Augmentation',
    linewidth=2
)

plt.plot(
    history_with.history['val_accuracy'],
    label='With Augmentation',
    linewidth=2
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_comparison.png")
plt.show()

# ============================================================
# Loss Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    history_without.history['val_loss'],
    label='Without Augmentation',
    linewidth=2
)

plt.plot(
    history_with.history['val_loss'],
    label='With Augmentation',
    linewidth=2
)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_comparison.png")
plt.show()

# ============================================================
# Save Best Model
# ============================================================

if best_with >= best_without:
    augmented_model.save("flowers_model.keras")
    best_model = "Augmented Model"
else:
    baseline_model.save("flowers_model.keras")
    best_model = "Baseline Model"

print("\nBest model saved as: flowers_model.keras")

# ============================================================
# Final Summary
# ============================================================

print("\n========================================")
print("DAY 17 TASK SUMMARY")
print("========================================")

print(f"Classes                     : {len(class_names)}")
print(f"Image Size                  : {IMAGE_SIZE}")
print(f"Batch Size                  : {BATCH_SIZE}")
print(f"Epochs                      : 10")

print("\nAugmentations Used:")
print("- RandomFlip")
print("- RandomRotation")
print("- RandomZoom")
print("- RandomContrast")
print("- ShiftScaleRotate (Albumentations)")
print("- HueSaturationValue (Albumentations)")

print("\nResults")
print("----------------------------------------")
print(f"Without Augmentation Accuracy : {best_without:.4f}")
print(f"With Augmentation Accuracy    : {best_with:.4f}")
print(f"Accuracy Gain                : {gain:.2f}%")
print(f"Best Model                  : {best_model}")

print("\nGenerated Files:")
print("✓ augmented_images.png")
print("✓ accuracy_comparison.png")
print("✓ loss_comparison.png")
print("✓ flowers_model.keras")

print("\nPart 3 Completed Successfully!")
print("Day 17 Task Completed Successfully!")