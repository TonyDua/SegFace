import cv2
import gradio as gr
from inference_single import load_model, preprocess_image, download_model_weights,postprocess_segmentation
from datasets.celebamask_hq import CelebAMaskHQ
from datasets.helen import HELEN
from datasets.lapa import LaPa
import torch
from PIL import Image

# 全局缓存模型
model_cache = {}

# 数据集映射关系
DATASET_CLASSES = {
    "celeba": CelebAMaskHQ,
    "lapa": LaPa,
    "helen": HELEN,
}


# 模型加载函数
def get_model_for_inference(model_name, resolution=512):
    """加载模型，缓存已加载的模型，避免重复加载"""
    if model_name not in model_cache:
        print(f"Loading model: {model_name}...")

        backbone = model_name.split("_")[0]  # 从模型名称推断 backbone 名称
        dataset = model_name.split("_")[1]  # 从模型名称推断数据集名称
        resolution = int(model_name.split("_")[2])  # 从模型名称推断分辨率

        model_path = download_model_weights(backbone,dataset,resolution)
        model = load_model(dataset=dataset, input_resolution=resolution,model_path=model_path)
        model_cache[model_name] = model
    return model_cache[model_name]


# 推理函数
def infer_image(model_name, editor_value):
    """
    1. 从 ImageEditor 组件获取裁剪后的图片。
    2. 调用推理逻辑，动态选择数据集类进行可视化。
    """
    # 获取数据集名称（从模型名称推断）
    if "celeba" in model_name:
        dataset_class = DATASET_CLASSES["celeba"]
    elif "lapa" in model_name:
        dataset_class = DATASET_CLASSES["lapa"]
    elif "helen" in model_name:
        dataset_class = DATASET_CLASSES["helen"]
    else:
        return "Unsupported dataset!"

    # 获取裁剪后的图像
    if editor_value is None or editor_value["composite"] is None:
        return "请先裁剪图片再进行推理！"

    cropped_image = editor_value["composite"]  # 获取裁剪结果
    cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)# 转换为 RGB 格式

    # 加载模型
    model = get_model_for_inference(model_name)

    # 预处理输入图像
    input_tensor, _, _ = preprocess_image(cropped_image, 512)
    input_tensor = input_tensor.cuda()

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor, labels=None, dataset=torch.tensor([1]).cuda())

    # 后处理与可视化
    mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    visualization = dataset_class.visualize_mask(mask)  # 使用已有的可视化方法
    return Image.fromarray(visualization)


# Gradio 界面设计
def create_gradio_interface():
    """创建 Gradio 界面，使用 ImageEditor 完成裁剪"""
    with gr.Blocks(title="SegFace 推理工具") as demo:
        gr.Markdown("## **SegFace 推理工具 - 使用裁剪功能**")

        with gr.Row():

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["convnext_celeba_512",
                             "efficientnet_celeba_512",
                             "mobilenet_celeba_512",
                             "resnet_celeba_512",
                             "swinb_celeba_224",
                             "swinb_celeba_256",
                             "swinb_celeba_448",
                             "swinb_celeba_512",
                             "swinb_lapa_224",
                             "swinb_lapa_256",
                             "swinb_lapa_448",
                             "swinb_lapa_512",
                             "swinv2b_celeba_512"],
                    label="选择模型",
                    value="swinb_celeba_512"
                )

            # 使用 ImageEditor 组件上传和裁剪图片
            with gr.Row():
                input_editor = gr.ImageEditor(
                    label="上传并裁剪图片",
                    sources=["upload"],
                    crop_size=(512, 512),
                    interactive=True
                )

            # 推理按钮
            with gr.Row():
                infer_button = gr.Button("执行推理")

        with gr.Column():
            # 显示推理结果
            output_image = gr.Image(type="pil", label="推理结果")

        # 绑定推理逻辑
        infer_button.click(infer_image, inputs=[model_dropdown, input_editor], outputs=output_image)

    return demo


# 启动 Gradio 服务
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
