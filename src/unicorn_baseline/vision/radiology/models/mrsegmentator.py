import json
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import (PlainConvEncoder,
                                                              PlainConvUNet)
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.utility.array import EnsureType


def remove_last_relu_encoder(model):
    last_relu_block = None

    # Iterate through encoder modules
    for m in model.modules():
        if hasattr(m, "nonlin") and isinstance(m.nonlin, nn.ReLU):
            last_relu_block = m

    if last_relu_block is not None:
        last_relu_block.nonlin = nn.Identity()
        # also fix inside `all_modules`
        if isinstance(last_relu_block.all_modules[-1], nn.ReLU):
            last_relu_block.all_modules[-1] = nn.Identity()
        print("Replaced last encoder ReLU with Identity.")
    else:
        print("No ReLU found in encoder!")


def load_model_mr(model_dir, fold: int = 0) -> PlainConvEncoder:
    with open(f"{model_dir}/plans.json") as f:
        plans = json.load(f)

    input_channels = 1  # len(dataset_json['channel_names']
    n_stages = len(
        plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"]
    )  # len(self.configuration["n_conv_per_stage_encoder"])
    features_per_stage = [
        min(
            plans["configurations"]["3d_fullres"]["UNet_base_num_features"] * 2**i,
            plans["configurations"]["3d_fullres"]["unet_max_num_features"],
        )
        for i in range(
            len(plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"])
        )
    ]  # self.UNet_max_features_3d = 320, n_stages
    conv_op = (
        nn.Conv3d
    )  # len(plans['configurations']['3d_fullres']['patch_size']) == 3 -> nn.Conv3d
    kernel_sizes = plans["configurations"]["3d_fullres"]["conv_kernel_sizes"]
    strides = plans["configurations"]["3d_fullres"]["pool_op_kernel_sizes"]
    n_conv_per_stage = plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"]
    num_classes = 41
    n_conv_per_stage_decoder = plans["configurations"]["3d_fullres"][
        "n_conv_per_stage_decoder"
    ]

    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={"inplace": True},
    )
    weights = torch.load(
        f"{model_dir}/fold_{fold}/checkpoint_final.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(weights["network_weights"])
    model = model.encoder
    remove_last_relu_encoder(model)
    return model


def load_data(data) -> DataLoader:
    train_transforms = Compose(
        [
            EnsureType(dtype=torch.float32),
            NormalizeIntensity(),
        ]
    )

    train_ds = Dataset(data=data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    return train_loader


def encode_mr(
    *,
    models: list[PlainConvEncoder],
    patch: sitk.Image,
    start_coord: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], tuple[int, int, int]]:
    """Encodes a given MR patch using an ensemble of convolutional encoders.

    This function takes a SimpleITK image patch, processes it through a list of
    encoder models, and extracts feature vectors from the deepest layer of each model.
    The outputs from the models are ensembled by concatenation. It then calculates
    the world coordinates for each feature vector in the resulting feature map and
    packages them into a list of dictionaries.

    Args:
        models: A list of PlainConvEncoder models to be used for encoding.
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

    # convert patch starting coordinate from physical to voxel index
    start_coord = patch.TransformPhysicalPointToIndex(start_coord)  # x, y, z

    # convert patch to numpy
    patch_array = sitk.GetArrayFromImage(patch)

    # expand patch to match encoder input requirements
    patch_array = np.expand_dims(patch_array, axis=(0, 1))

    ensemble_outputs = []
    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            train_loader = load_data(patch_array)
            input = next(iter(train_loader))
            input = input.to(device)
            output: list[torch.Tensor] = model(input)

        output = output[-1]  # select deepest layer

        # convert to numpy
        output = output.cpu().detach().numpy()  # z, y, x
        ensemble_outputs.append(output)

    # ensemble the outputs
    ensemble_output = np.concatenate(ensemble_outputs, axis=1)
    print(f"Ensembled output shape: {ensemble_output.shape}")

    # create sub-patch coordinates
    stride = (2**5, 2**5, 2**4)  # x, y, z
    sub_coords = []
    sub_outputs = []
    for x_step in range(ensemble_output.shape[-1]):
        for y_step in range(ensemble_output.shape[-2]):
            for z_step in range(ensemble_output.shape[-3]):
                features = ensemble_output[..., z_step, y_step, x_step].squeeze()
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
