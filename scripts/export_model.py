import argparse
import torch

def export_to_onnx(model, dummy_image, output_path="model.onnx"):
    torch.onnx.export(
        model,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['images'],
        output_names=['caption_logits']
    )
    print(f"Exported model to {output_path}")

if __name__ == '__main__':
    print("ONNX/TorchScript export utility ready.")
