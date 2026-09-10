import torch
from torch.utils.data import DataLoader

from dataset import SemanticKITTIDataset
from model import PointNetSegmentation


# ============================================================
# DATASET
# ============================================================

dataset = SemanticKITTIDataset(
    root_dir="semantic_kitti_sample/dataset",
    sequence="00",
    frames=[0, 1, 2, 3, 4],
    max_points=50000,
    normalize=False
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = PointNetSegmentation(
    input_features=4,
    num_classes=25
)

model.eval()


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING DATASET + MODEL")
print("=" * 60)


with torch.no_grad():

    for batch in loader:

        print("\nNumber of items returned by dataset:",
              len(batch))

        features = batch[0]
        training_labels = batch[1]
        raw_labels = batch[2]
        frame = batch[3]

        print("Frame:", frame)

        print(
            "Input features:",
            features.shape
        )

        print(
            "Training labels:",
            training_labels.shape
        )

        print(
            "Raw labels:",
            raw_labels.shape
        )

        # ----------------------------------------------------
        # MODEL FORWARD PASS
        # ----------------------------------------------------

        logits = model(features)

        print(
            "Model output:",
            logits.shape
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        print(
            "Predictions:",
            predictions.shape
        )

        # ----------------------------------------------------
        # CHECK
        # ----------------------------------------------------

        assert features.shape == (
            1,
            50000,
            4
        )

        assert training_labels.shape == (
            1,
            50000
        )

        assert logits.shape == (
            1,
            50000,
            25
        )

        assert predictions.shape == (
            1,
            50000
        )

        print("\n✓ Dataset + model test successful!")

        break