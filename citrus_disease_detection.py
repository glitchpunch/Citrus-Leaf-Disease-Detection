"""
Citrus Leaf Disease Detection - Base CNN + Transfer Learning + Grad-CAM
Dataset: https://www.kaggle.com/datasets/superlord/citrus-diseases
"""

import os
import random
import subprocess

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from tensorflow.keras.applications import mobilenet_v2, efficientnet, resnet_v2

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, auc,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

import splitfolders

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RAW_DIR = "citrus_raw"
SPLIT_DIR = "citrus_split"
OUTPUT_DIR = "outputs"
SPLIT_RATIO = (0.7, 0.2, 0.1)

EPOCHS_FROZEN = 8
EPOCHS_FINETUNE = 5
EPOCHS_BASE_CNN = 20
UNFREEZE_LAST_N = 40
LR_FROZEN = 1e-4
LR_FINETUNE = 1e-5
LR_BASE_CNN = 1e-3

RUN_GRADCAM = True
GRADCAM_SAMPLES = 3

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_dataset_if_needed():
    if os.path.isdir(RAW_DIR) and os.listdir(RAW_DIR):
        return
    print("Raw dataset not found, attempting Kaggle download...")
    zip_name = "citrus-diseases.zip"
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "superlord/citrus-diseases", "-p", "."],
            check=True,
        )
        os.makedirs(RAW_DIR, exist_ok=True)
        subprocess.run(["unzip", "-oq", zip_name, "-d", RAW_DIR], check=True)
    except Exception as exc:
        raise RuntimeError(
            "Automatic Kaggle download failed. Download the dataset manually from "
            "https://www.kaggle.com/datasets/superlord/citrus-diseases, unzip it, and "
            f"place the contents inside a folder named '{RAW_DIR}' next to this script."
        ) from exc


def find_class_root(root_dir):
    image_exts = (".jpg", ".jpeg", ".png")
    for current, dirs, _ in os.walk(root_dir):
        if len(dirs) < 2:
            continue
        sub_has_images = all(
            any(f.lower().endswith(image_exts) for f in os.listdir(os.path.join(current, d)))
            for d in dirs
        )
        if sub_has_images:
            return current
    raise RuntimeError(f"Could not locate class folders inside '{root_dir}'.")


def prepare_dataset():
    train_dir = os.path.join(SPLIT_DIR, "train")
    if os.path.isdir(train_dir):
        return
    download_dataset_if_needed()
    class_root = find_class_root(RAW_DIR)
    print(f"Detected class folders in: {class_root}")
    splitfolders.ratio(class_root, output=SPLIT_DIR, seed=SEED, ratio=SPLIT_RATIO)


PREPROCESS_FUNCS = {
    "mobilenetv2": mobilenet_v2.preprocess_input,
    "efficientnetb0": efficientnet.preprocess_input,
    "resnet50v2": resnet_v2.preprocess_input,
}


def make_generators(preprocess_input=None):
    common_aug = dict(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    if preprocess_input is None:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, **common_aug)
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255)
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **common_aug)
        eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "train"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "val"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


def build_base_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="base_cnn")


TL_BACKBONES = {
    "mobilenetv2": MobileNetV2,
    "efficientnetb0": EfficientNetB0,
    "resnet50v2": ResNet50V2,
}


def build_transfer_model(name, input_shape, num_classes):
    backbone_cls = TL_BACKBONES[name]
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs, name=name)
    return model, base

def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, f"{model_name}_best.h5"),
                                   monitor="val_loss", save_best_only=True),
    ]


def train_base_cnn(train_gen, val_gen, num_classes):
    model = build_base_cnn(IMG_SIZE + (3,), num_classes)
    model.compile(optimizer=optimizers.Adam(LR_BASE_CNN),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    class_weights = get_class_weights(train_gen)
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_BASE_CNN,
                         class_weight=class_weights, callbacks=get_callbacks("base_cnn"))
    return model, history.history


def train_transfer_model(name, train_gen, val_gen, num_classes):
    model, base = build_transfer_model(name, IMG_SIZE + (3,), num_classes)
    class_weights = get_class_weights(train_gen)

    model.compile(optimizer=optimizers.Adam(LR_FROZEN),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FROZEN,
                          class_weight=class_weights, callbacks=get_callbacks(name))

    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(LR_FINETUNE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINETUNE,
                          class_weight=class_weights, callbacks=get_callbacks(name))

    combined_history = {
        "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
        "loss": history1.history["loss"] + history2.history["loss"],
        "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
    }
    return model, combined_history


def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="Train")
    axes[0].plot(history["val_accuracy"], label="Val")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Val")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_training_curves.png"))
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, model_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close(fig)


def plot_roc_curves(y_true_onehot, y_pred_proba, class_names, model_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cname} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_roc_curve.png"))
    plt.close(fig)


def specificity_macro(cm):
    total = cm.sum()
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def evaluate_model(model, test_gen, model_name):
    test_gen.reset()
    y_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    if len(class_names) == 2:
        y_true_onehot = np.eye(2)[y_true]
    else:
        y_true_onehot = label_binarize(y_true, classes=range(len(class_names)))

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall/Sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Specificity": specificity_macro(cm),
        "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true_onehot, y_proba, average="macro", multi_class="ovr"),
    }

    plot_confusion_matrix(cm, class_names, model_name)
    plot_roc_curves(y_true_onehot, y_proba, class_names, model_name)

    with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return metrics


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_layer, pred_index=None):
    grad_model = models.Model([model.inputs], [model.get_layer(last_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap


def save_gradcam_overlay(img_path, heatmap, out_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_gradcam_demo(model, model_name, test_gen, preprocess_input, num_samples=GRADCAM_SAMPLES):
    last_layer = find_last_conv_layer(model)
    if last_layer is None:
        print(f"[{model_name}] No 4D conv layer found, skipping Grad-CAM.")
        return
    sample_paths = random.sample(test_gen.filepaths, min(num_samples, len(test_gen.filepaths)))
    for i, path in enumerate(sample_paths):
        img = keras_image.load_img(path, target_size=IMG_SIZE)
        arr = keras_image.img_to_array(img)
        arr = arr / 255.0 if preprocess_input is None else preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        try:
            heatmap = make_gradcam_heatmap(arr, model, last_layer)
            out_path = os.path.join(OUTPUT_DIR, f"{model_name}_gradcam_{i}.png")
            save_gradcam_overlay(path, heatmap, out_path)
        except Exception as exc:
            print(f"[{model_name}] Grad-CAM failed on sample {i}: {exc}")


def main():
    prepare_dataset()
    results = []

    train_gen, val_gen, test_gen = make_generators(preprocess_input=None)
    num_classes = train_gen.num_classes
    print("Classes:", train_gen.class_indices)

    print("\n=== Training base_cnn ===")
    base_model, base_history = train_base_cnn(train_gen, val_gen, num_classes)
    plot_training_curves(base_history, "base_cnn")
    results.append(evaluate_model(base_model, test_gen, "base_cnn"))
    base_model.save(os.path.join(OUTPUT_DIR, "base_cnn_final.h5"))
    if RUN_GRADCAM:
        run_gradcam_demo(base_model, "base_cnn", test_gen, preprocess_input=None)

    for name in TL_BACKBONES:
        print(f"\n=== Training {name} ===")
        preprocess_input = PREPROCESS_FUNCS[name]
        train_gen, val_gen, test_gen = make_generators(preprocess_input=preprocess_input)
        model, history = train_transfer_model(name, train_gen, val_gen, num_classes)
        plot_training_curves(history, name)
        results.append(evaluate_model(model, test_gen, name))
        model.save(os.path.join(OUTPUT_DIR, f"{name}_final.h5"))
        if RUN_GRADCAM:
            run_gradcam_demo(model, name, test_gen, preprocess_input=preprocess_input)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"), index=False)
    print("\n=== Final Comparison ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
