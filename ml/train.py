import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .config import (
    CANDIDATE_DIR,
    MODEL_DIR,
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
)

from .model import build_model, freeze_backbone
from .preprocess import get_train_transform, get_eval_transform



# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# Classes

CLASSES = sorted(
    [p.name for p in CANDIDATE_DIR.iterdir() if p.is_dir()]
)

CLASS_TO_IDX = {
    cls: i
    for i, cls in enumerate(CLASSES)
}

IDX_TO_CLASS = {
    i: cls
    for cls, i in CLASS_TO_IDX.items()
}



# Dataset

class TileDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")

        image = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]

        return image, label



# Build dataframe

def build_dataframe():

    records = []

    for cls in CLASSES:

        class_dir = CANDIDATE_DIR / cls

        for image_path in sorted(class_dir.glob("*.png")):

            records.append({
                "path": str(image_path),
                "label": cls,
            })

    df = pd.DataFrame(records)

    print(f"Total candidate tiles: {len(df)}")
    print(df["label"].value_counts())

    return df



# Build dataloaders

def build_dataloaders(df):

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=SEED,
    )

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")

    train_dataset = TileDataset(
        train_df,
        get_train_transform(IMG_SIZE),
    )

    val_dataset = TileDataset(
        val_df,
        get_eval_transform(IMG_SIZE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader



# Train one epoch

def run_epoch(model, loader, criterion, optimizer= None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:

                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            total_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            total_samples += images.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )



# Save model artifacts

def save_artifacts(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    # Model weights
    torch.save(
        model.state_dict(),
        MODEL_DIR / "model.pt",
    )

    
    # Class mapping
    with open(
        MODEL_DIR / "class_mapping.json",
        "w",
    ) as f:

        json.dump(
            {
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
            },
            f,
            indent=2,
        )

    
    # Model configuration
    model_config = {

        "model_type":
            "mobilenet_v3_small_transfer",

        "img_size":
            IMG_SIZE,

        "normalize_mean":
            [0.485, 0.456, 0.406],

        "normalize_std":
            [0.229, 0.224, 0.225],

        "model_version":
            "mobilenetv3-small-v1",

    }

    with open(
        MODEL_DIR / "model_config.json",
        "w",
    ) as f:

        json.dump(
            model_config,
            f,
            indent=2,
        )

    print()
    print("Model artifacts saved.")
    print(f"  Weights: {MODEL_DIR / 'model.pt'}")
    print(f"  Classes: {MODEL_DIR / 'class_mapping.json'}")
    print(f"  Config:  {MODEL_DIR / 'model_config.json'}")



# Main training function

def train_model():

    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    
    # Dataset
    df = build_dataframe()

    train_loader, val_loader = build_dataloaders(df)

    
    # Model
    model = build_model(
        num_classes=len(CLASSES),
        pretrained=True,
    )

    model = freeze_backbone(model)

    model = model.to(DEVICE)

    
    # Optimizer
    trainable_params = [
        p
        for p in model.classifier.parameters()
        if p.requires_grad
    ]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        trainable_params,
        lr=LEARNING_RATE,
    )

    
    # Training loop
    
    # NOTE: I am saving the best weights during training and then calling save_artifacts(model) at the end, 
    # Problem: 'model.pt' could get overwritten by the final epoch. 
    # Solution: I will save the best state in memory and save it once at the end via save_artifacts():

    best_state_dict = None
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }


    # Restore best model
    model.load_state_dict(best_state_dict)

    save_artifacts(model)

    return model