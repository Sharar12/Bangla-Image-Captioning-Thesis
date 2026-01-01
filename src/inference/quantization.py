import torch

def quantize_decoder_dynamic(model):
    """Applies PyTorch dynamic INT8 quantization to decoder Linear and Recurrent layers."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.GRUCell, torch.nn.LSTMCell}, dtype=torch.qint8
    )
    return quantized_model
