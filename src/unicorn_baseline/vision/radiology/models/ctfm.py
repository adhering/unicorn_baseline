from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from lighter_zoo import SegResNet
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.utility.array import EnsureType


def load_model_ct(model_path: Path | str) -> SegResNet:
    # Load pre-trained model
    return SegResNet.from_pretrained(model_path)


def load_data(data) -> DataLoader:
    train_transforms = Compose(
        [
            EnsureType(),
            ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        ]
    )

    train_ds = Dataset(data=data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    return train_loader


def encode_ct(
    *,
    model: SegResNet,
    patch: sitk.Image,
    start_coord: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], tuple[int, int, int]]:
    """Encodes a given CT patch using a convolutional encoder.

    This function takes a SimpleITK image patch, processes it through the encoder model,
    and extracts feature vectors from the deepest layer of the model. It then calculates
    the world coordinates for each feature vector in the resulting feature map and
    packages them into a list of dictionaries.

    Args:
        model: A SegResNet model to be used for encoding.
        patch: A SimpleITK Image object representing the patch to be encoded.
        start_coord: The physical world coordinate (x, y, z) of the starting
            point of the patch.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dictionary holds the 'coordinates'
          (as a tuple of floats) and 'features' (as a numpy array) for a
          sub-patch.
        - A tuple representing the stride (x, y, z) used to calculate the
          sub-patch coordinates, corresponding to the model's downsampling factor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # convert patch starting coordinate from physical to voxel index
    start_coord = patch.TransformPhysicalPointToIndex(start_coord)  # x, y, z

    # convert patch to numpy
    patch_array = sitk.GetArrayFromImage(patch)

    # expand patch to match encoder input requirements
    patch_array = np.expand_dims(patch_array, axis=(0, 1))

    model.eval()
    with torch.no_grad():
        train_loader = load_data(patch_array)
        input = next(iter(train_loader))
        input = input.to(device)
        model_output = model.encoder(input)

    final_layer_output: torch.Tensor = model_output[-1]  # select deepest layer

    # convert to numpy
    output = final_layer_output.cpu().detach().numpy()

    # extract features from each spatial location (sub-patch approach)
    stride = (2**4, 2**4, 2**4)  # x, y, z
    sub_coords = []
    sub_outputs = []
    for x_step in range(output.shape[-1]):
        for y_step in range(output.shape[-2]):
            for z_step in range(output.shape[-3]):
                features = output[..., z_step, y_step, x_step].squeeze()
                coord = (
                    start_coord[0] + x_step * stride[0],
                    start_coord[1] + y_step * stride[1],
                    start_coord[2] + z_step * stride[2],
                )
                sub_outputs.append(features)
                sub_coords.append(coord)

    # convert sub coordinates to world coordines
    sub_coords = [
        patch.TransformIndexToPhysicalPoint(coord)
        for coord in sub_coords
    ]

    # create patch_features
    patch_features = [
        {
            "coordinates": coord,
            "features": features,
        }
        for coord, features in zip(sub_coords, sub_outputs)
    ]

    return patch_features, stride
