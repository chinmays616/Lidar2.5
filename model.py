# ============================================================
# model.py
# PointNet-style Semantic Segmentation for SemanticKITTI
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 25
INPUT_FEATURES = 4       # x, y, z, intensity


# ============================================================
# SHARED MLP BLOCK
# ============================================================

class SharedMLP(nn.Module):
    """
    Applies the same MLP independently to every point.

    Input:
        [B, N, C_in]

    Output:
        [B, N, C_out]
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                in_channels,
                out_channels
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # x:
        # [B, N, C]

        B, N, C = x.shape

        # Linear expects:
        # [B*N, C]

        x = x.reshape(
            B * N,
            C
        )

        x = self.mlp(x)

        # Restore:
        # [B, N, C_out]

        x = x.reshape(
            B,
            N,
            -1
        )

        return x


# ============================================================
# POINTNET SEGMENTATION NETWORK
# ============================================================

class PointNetSegmentation(nn.Module):
    """
    PointNet-style semantic segmentation network.

    Input:
        [B, N, 4]

    Features:
        x, y, z, intensity

    Output:
        [B, N, NUM_CLASSES]

    Each point receives a prediction over all
    SemanticKITTI training classes.
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        self.input_features = input_features
        self.num_classes = num_classes

        # ----------------------------------------------------
        # POINT FEATURE EXTRACTION
        # ----------------------------------------------------

        self.mlp1 = SharedMLP(
            input_features,
            64
        )

        self.mlp2 = SharedMLP(
            64,
            128
        )

        self.mlp3 = SharedMLP(
            128,
            256
        )

        # ----------------------------------------------------
        # GLOBAL FEATURE
        # ----------------------------------------------------

        self.global_mlp = SharedMLP(
            256,
            512
        )

        # ----------------------------------------------------
        # SEGMENTATION HEAD
        # ----------------------------------------------------

        # Local features:
        # 64 + 128 + 256
        #
        # Global feature:
        # 512
        #
        # Total:
        # 64 + 128 + 256 + 512 = 960

        segmentation_input = (
            64 +
            128 +
            256 +
            512
        )

        self.segmentation_mlp1 = SharedMLP(
            segmentation_input,
            256
        )

        self.segmentation_mlp2 = SharedMLP(
            256,
            128
        )

        # Final classifier
        #
        # No softmax here.
        #
        # CrossEntropyLoss expects raw logits.

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        """
        Parameters
        ----------
        x : torch.Tensor

            Shape:
                [B, N, 4]

            where:
                B = batch size
                N = number of points
                4 = x,y,z,intensity

        Returns
        -------
        logits : torch.Tensor

            Shape:
                [B, N, num_classes]
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if x.ndim != 3:

            raise ValueError(
                "Expected input shape "
                "[B, N, C], "
                f"but received {x.shape}"
            )

        if x.shape[-1] != self.input_features:

            raise ValueError(
                f"Expected {self.input_features} "
                f"input features, "
                f"but received {x.shape[-1]}"
            )

        # ----------------------------------------------------
        # Local feature extraction
        # ----------------------------------------------------

        feature_64 = self.mlp1(x)

        # [B, N, 64]

        feature_128 = self.mlp2(
            feature_64
        )

        # [B, N, 128]

        feature_256 = self.mlp3(
            feature_128
        )

        # [B, N, 256]

        # ----------------------------------------------------
        # Global feature
        # ----------------------------------------------------

        global_feature = self.global_mlp(
            feature_256
        )

        # [B, N, 512]

        # ----------------------------------------------------
        # Global max pooling
        # ----------------------------------------------------

        global_feature = torch.max(
            global_feature,
            dim=1,
            keepdim=True
        )[0]

        # [B, 1, 512]

        # ----------------------------------------------------
        # Broadcast global feature to every point
        # ----------------------------------------------------

        num_points = x.shape[1]

        global_feature = global_feature.expand(
            -1,
            num_points,
            -1
        )

        # [B, N, 512]

        # ----------------------------------------------------
        # Combine local + global information
        # ----------------------------------------------------

        combined_features = torch.cat(
            [
                feature_64,
                feature_128,
                feature_256,
                global_feature
            ],
            dim=-1
        )

        # [B, N, 960]

        # ----------------------------------------------------
        # Segmentation head
        # ----------------------------------------------------

        x = self.segmentation_mlp1(
            combined_features
        )

        # [B, N, 256]

        x = self.segmentation_mlp2(
            x
        )

        # [B, N, 128]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(x)

        # [B, N, 25]

        return logits


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING POINTNET SEMANTIC SEGMENTATION MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = PointNetSegmentation(
        input_features=4,
        num_classes=25
    )

    print("\nModel created successfully.")

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Create dummy LiDAR batch
    # --------------------------------------------------------

    batch_size = 1
    num_points = 50000
    num_features = 4

    dummy_input = torch.randn(
        batch_size,
        num_points,
        num_features
    )

    print(
        "\nInput shape:",
        dummy_input.shape
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            dummy_input
        )

    print(
        "Output shape:",
        logits.shape
    )

    # --------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        num_points,
        NUM_CLASSES
    )

    print(
        "Expected shape:",
        expected_shape
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    assert logits.shape == expected_shape, (
        f"Shape mismatch! "
        f"Got {logits.shape}, "
        f"expected {expected_shape}"
    )

    print("\n✓ Forward pass successful.")

    # --------------------------------------------------------
    # Convert logits to predictions
    # --------------------------------------------------------

    predictions = torch.argmax(
        logits,
        dim=-1
    )

    print(
        "\nPrediction shape:",
        predictions.shape
    )

    print(
        "Expected prediction shape:",
        (batch_size, num_points)
    )

    # --------------------------------------------------------
    # Check prediction range
    # --------------------------------------------------------

    print(
        "\nMinimum predicted class:",
        predictions.min().item()
    )

    print(
        "Maximum predicted class:",
        predictions.max().item()
    )

    assert predictions.min() >= 0
    assert predictions.max() < NUM_CLASSES

    print(
        "\n✓ Predictions are within "
        f"0-{NUM_CLASSES - 1}."
    )

    # --------------------------------------------------------
    # Test CrossEntropyLoss
    # --------------------------------------------------------

    dummy_labels = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(
            batch_size,
            num_points
        ),
        dtype=torch.long
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=255
    )

    # CrossEntropyLoss expects:

    # logits:
    # [B, C, N]

    # labels:
    # [B, N]

    loss = criterion(
        logits.permute(0, 2, 1),
        dummy_labels
    )

    print(
        "\nCrossEntropyLoss:",
        loss.item()
    )

    print(
        "\n✓ Loss calculation successful."
    )

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
# import torch

# from torch.utils.data import DataLoader


# dataset = SemanticKITTIDataset(
#     root_dir="semantic_kitti_sample/dataset",
#     sequence="00",
#     frames=[0, 1, 2, 3, 4],
#     max_points=50000,
# )

# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0
# )


# model = PointNetSegmentation(
#     input_features=4,
#     num_classes=25
# )


# model.eval()


# with torch.no_grad():


#         print("Frame:", frame)

#         print(
#             "Input:",
#             features.shape
#         )

#         logits = model(features)

#         print(
#             "Logits:",
#             logits.shape
#         )

#         predictions = torch.argmax(
#             logits,
#             dim=-1
#         )

#         print(
#             "Predictions:",
#             predictions.shape
#         )

#         break