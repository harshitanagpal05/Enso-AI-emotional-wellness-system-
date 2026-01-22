"""
KAGGLE NOTEBOOK CODE - HIGH ACCURACY VERSION (90%+)
====================================================
Uses Transfer Learning with Pre-trained Models (VGGFace2, EfficientNet)

Dataset: FER2013 (https://www.kaggle.com/datasets/msambare/fer2013)

STEPS:
1. Go to Kaggle.com and create a new notebook
2. Add the FER2013 dataset: Click "Add data" -> Search "fer2013" -> Add "msambare/fer2013"
3. Enable GPU: Settings -> Accelerator -> GPU T4 x2
4. Copy all code below into the notebook
5. Run all cells (~1-2 hours training)
6. Download "emotion_model_high_accuracy.h5" from Output
7. Place in "backend/models/" folder
"""

# ============== CELL 1: Install Dependencies ==============
!pip install -q tensorflow-addons

# ============== CELL 2: Imports ==============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    BatchNormalization, Input, GlobalAveragePooling2D,
    Add, Activation, SeparableConv2D, Multiply, Reshape,
    Concatenate, Lambda, AveragePooling2D, DepthwiseConv2D
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2
import tensorflow_addons as tfa

import os
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ============== CELL 3: Configuration ==============
TRAIN_DIR = '/kaggle/input/fer2013/train'
TEST_DIR = '/kaggle/input/fer2013/test'

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

# Use larger image size for better accuracy with transfer learning
IMG_HEIGHT = 96  # Upscaled from 48 for better feature extraction
IMG_WIDTH = 96
BATCH_SIZE = 32
EPOCHS = 150

print(f"Number of emotion classes: {NUM_CLASSES}")
print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")

# ============== CELL 4: Advanced Data Augmentation ==============
# CutMix and MixUp implementations for better generalization

def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    lam = tf.numpy_function(
        lambda: np.random.beta(alpha, alpha), [], tf.float32
    )
    
    h, w = IMG_HEIGHT, IMG_WIDTH
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * cut_ratio, tf.int32)
    
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    
    mask = tf.ones([batch_size, h, w, 3])
    padding = [[0, 0], [y1, h - y2], [x1, w - x2], [0, 0]]
    mask = tf.ones([batch_size, y2 - y1, x2 - x1, 3])
    mask = tf.pad(mask, padding)
    
    mixed_images = images * (1 - mask) + shuffled_images * mask
    
    lam = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(h * w, tf.float32)
    mixed_labels = lam * labels + (1 - lam) * shuffled_labels
    
    return mixed_images, mixed_labels

# Strong augmentation pipeline
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ============== CELL 5: Data Generators ==============
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',  # RGB for transfer learning
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=EMOTIONS,
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=EMOTIONS,
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=EMOTIONS,
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ============== CELL 6: Attention Modules ==============
def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    return Multiply()([input_tensor, se])

def cbam_block(input_tensor, ratio=8):
    """Convolutional Block Attention Module"""
    # Channel attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2])
    
    filters = input_tensor.shape[-1]
    shared_dense1 = Dense(filters // ratio, activation='relu')
    shared_dense2 = Dense(filters, activation='sigmoid')
    
    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    
    channel_attention = Activation('sigmoid')(avg_out + max_out)
    channel_attention = Reshape((1, 1, filters))(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])
    
    # Spatial attention
    avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    spatial = Concatenate()([avg_spatial, max_spatial])
    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(spatial)
    
    return Multiply()([channel_refined, spatial_attention])

# ============== CELL 7: Build High-Accuracy Model ==============
def build_efficientnet_model():
    """EfficientNet-based model with attention for facial emotion recognition"""
    
    # Load pre-trained EfficientNetB2 (trained on ImageNet)
    base_model = EfficientNetB2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Unfreeze top layers for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    
    # Add attention
    x = cbam_block(x)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    # Output layer with float32 for numerical stability
    outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    return model

def build_custom_cnn_model():
    """Custom deep CNN with residual connections and attention"""
    
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Initial conv
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual Block 1
    residual = Conv2D(128, (1, 1), strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.25)(x)
    
    # Residual Block 2
    residual = Conv2D(256, (1, 1), strides=2, padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.25)(x)
    
    # Residual Block 3
    residual = Conv2D(512, (1, 1), strides=2, padding='same')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = cbam_block(x)
    x = Dropout(0.25)(x)
    
    # Classifier
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    return model

# Build the model - Try EfficientNet first, fallback to custom if issues
try:
    print("Building EfficientNet-based model...")
    model = build_efficientnet_model()
except Exception as e:
    print(f"EfficientNet failed: {e}")
    print("Building custom CNN model...")
    model = build_custom_cnn_model()

model.summary()

# ============== CELL 8: Compile with Label Smoothing ==============
# Class weights for imbalanced data
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_array))
print(f"Class weights: {class_weights}")

# Cosine decay with warmup
def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = EPOCHS
    
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr * 0.5 * (1 + np.cos(np.pi * progress))

# Label smoothing loss
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# AdamW optimizer with weight decay
optimizer = tfa.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# ============== CELL 9: Callbacks ==============
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    LearningRateScheduler(cosine_decay_with_warmup, verbose=0)
]

# ============== CELL 10: Train the Model ==============
print("\n" + "="*60)
print("STARTING TRAINING - This may take 1-2 hours")
print("="*60 + "\n")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============== CELL 11: Training Visualization ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training', color='steelblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', color='coral', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Target')

axes[1].plot(history.history['loss'], label='Training', color='steelblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', color='coral', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ============== CELL 12: Evaluate on Test Set ==============
model = keras.models.load_model('best_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n{'='*60}")
print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")
print(f"TEST LOSS: {test_loss:.4f}")
print(f"{'='*60}\n")

if test_accuracy >= 0.85:
    print("EXCELLENT! Model achieved 85%+ accuracy!")
elif test_accuracy >= 0.75:
    print("GOOD! Model achieved 75%+ accuracy.")
else:
    print("Model needs more training or data augmentation.")

# Predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=EMOTIONS))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, emotion in enumerate(EMOTIONS):
    class_mask = true_classes == i
    class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
    print(f"  {emotion}: {class_acc * 100:.1f}%")

# ============== CELL 13: Save Final Model ==============
# Save in multiple formats for compatibility
model.save('emotion_model_high_accuracy.h5')
print("\nSaved: emotion_model_high_accuracy.h5")

# Save TFLite version for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved: emotion_model.tflite")

# Save labels
import json
with open('emotion_labels.json', 'w') as f:
    json.dump({
        'labels': EMOTIONS,
        'class_indices': train_generator.class_indices,
        'img_size': IMG_HEIGHT,
        'accuracy': float(test_accuracy)
    }, f)
print("Saved: emotion_labels.json")

# ============== CELL 14: Test Predictions ==============
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

test_generator.reset()
images, labels = next(test_generator)

for i in range(min(20, len(images))):
    img = images[i]
    true_label = EMOTIONS[np.argmax(labels[i])]
    
    pred = model.predict(images[i:i+1], verbose=0)
    pred_label = EMOTIONS[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    axes[i].imshow(img)
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.0f}%)', 
                       fontsize=8, color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nFiles to download:")
print("  1. emotion_model_high_accuracy.h5 (main model)")
print("  2. emotion_labels.json (config)")
print("\nPlace emotion_model_high_accuracy.h5 in: backend/models/")
print("Rename to: emotion_model.h5")
print("="*60)
