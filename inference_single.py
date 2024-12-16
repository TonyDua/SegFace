import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download
from network import get_model

from datasets.celebamask_hq import CelebAMaskHQ
from datasets.helen import HELEN
from datasets.lapa import LaPa

def download_model_weights(model_name, dataset, resolution):
    """Automatically download the model weights based on user selection."""
    repo_id = "kartiknarayan/SegFace"
    filename = f"{model_name}_{dataset}_{resolution}/model_299.pt"
    local_dir = "./weights"
    os.makedirs(local_dir, exist_ok=True)

    # Download the model weight file
    print(f"Downloading weights for {model_name} trained on {dataset} at {resolution} resolution...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(f"Model weights downloaded to {model_path}")
    return model_path


def preprocess_image(image_path, input_resolution):
    """Load, resize, pad, and preprocess an image to target resolution."""
    if isinstance(image_path,np.ndarray):
        image = image_path
    else:
        image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    image = image[:, :, ::-1].copy()  # BGR to RGB，添加 .copy() 确保内存连续
    original_h, original_w = image.shape[:2]

    # 情况1：图像正好是 input_resolution
    if original_h == input_resolution and original_w == input_resolution:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0), (0, 0, 0, 0), (original_h, original_w)

    # 情况2：等比例缩放，长边= input_resolution
    scale = input_resolution / max(original_h, original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 添加Padding到 input_resolution
    top = (input_resolution - new_h) // 2
    bottom = input_resolution - new_h - top
    left = (input_resolution - new_w) // 2
    right = input_resolution - new_w - left
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 转换为Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_padded).unsqueeze(0), (top, bottom, left, right), (original_h, original_w)


def postprocess_segmentation(mask, padding, original_size):
    """Remove padding and resize the mask back to the original size."""
    top, bottom, left, right = padding
    original_h, original_w = original_size

    # 去除Padding
    mask_cropped = mask[top: mask.shape[0] - bottom, left: mask.shape[1] - right]

    # 插值回原始图像大小
    mask_resized = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized



def load_model(dataset, input_resolution, model_path):
    """Load the pretrained model."""
    # Map the user-friendly backbone name to the expected format
    if "celeba" in dataset:
        model_backbone = "segface_celeb"
    elif "lapa" in dataset:
        model_backbone = "segface_lapa"
    elif "helen" in dataset:
        model_backbone = "segface_helen"
    else:
        raise ValueError("Unsupported backbone or dataset")

    # Initialize and load the model
    if "swinb" in model_path:
        model_name = "swin_base"
    elif "convnext" in model_path:
        model_name = "convnext_base"
    elif "efficientnet" in model_path:
        model_name = "efficientnet"
    elif "mobilenet" in model_path:
        model_name = "mobilenet_v3_small"
    elif "resnet" in model_path:
        model_name = "resnet"
    elif "swinv2b" in model_path:
        model_name = "swinv2_base"

    model = get_model(backbone=model_backbone,
                      input_resolution = input_resolution,
                      model=model_name).cuda()
    model.eval()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    return model


def infer_single_image(model, image_tensor, input_resolution):
    """Perform inference on a single image."""
    with torch.no_grad():
        # Placeholder for labels and dataset since they are not used in inference
        labels = {"lnm_seg": torch.zeros(1, 5, 2).cuda()}  # Dummy landmarks
        dataset = torch.tensor([1]).cuda()  # Dummy dataset indicator

        # Forward pass through the model
        seg_output = model(image_tensor, labels, dataset)

        # Resize output back to original resolution
        mask = F.interpolate(seg_output, size=(input_resolution, input_resolution), mode='bilinear', align_corners=False)
        mask = mask.softmax(dim=1)  # Convert logits to probabilities
        preds = torch.argmax(mask, dim=1)  # Get the class with the highest probability
        return preds.cpu().numpy()[0]  # Remove batch dimension


def visualize_segmentation(dataset_class, mask, save_path=None):
    """Visualize the segmentation mask using dataset's static method."""
    visualization = dataset_class.visualize_mask(mask)
    if save_path:
        cv2.imwrite(save_path, visualization[:, :, ::-1])  # Save as BGR
    return visualization


if __name__ == "__main__":
    # User configuration
    image_path = "test_image.png"  # Path to the input image
    backbone = "swinb"  # Model backbone: swinb, convnext, mobilenet, etc.
    dataset = "celeba"  # Dataset the model was trained on: celeba or lapa
    resolution = 512  # Resolution: 224, 256, 448, or 512
    output_path = "segmentation_result_512.png"  # Output path for the visualization

    # Download model weights
    model_path = download_model_weights(backbone, dataset, resolution)

    # 图像预处理
    image_tensor, padding, original_size = preprocess_image(image_path, resolution)
    image_tensor = image_tensor.cuda()


    # Load model
    model = load_model(f"{backbone}_{dataset}", resolution, model_path)

    # Perform inference
    mask = infer_single_image(model, image_tensor, resolution)

    # Choose dataset class based on dataset
    dataset_class = {
        "celeba": CelebAMaskHQ,
        "lapa": LaPa,
        "helen": HELEN,
    }[dataset]

    # 直接保存模型输出（512x512）
    mask_output_path = "model_output_512x512.png"
    visualization = dataset_class.visualize_mask(mask)
    cv2.imwrite(mask_output_path, visualization[:, :, ::-1])
    print(f"Model output saved to {mask_output_path}")

    # 后处理，去除Padding并恢复到原始图像大小
    mask_resized = postprocess_segmentation(mask, padding, original_size)



    visualization = dataset_class.visualize_mask(mask_resized)
    cv2.imwrite(output_path, visualization[:, :, ::-1])
    print(f"Segmentation result saved to {output_path}")