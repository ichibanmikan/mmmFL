import torch

# 检查是否支持 bfloat8 和 float8_e4m3/e5m2 精度
print("Supports float8_e4m3: ", torch.cuda.is_bf16_supported())  # 检查支持情况
print("Supports float8_e5m2: ", torch.cuda.is_fp8_supported())  # PyTorch 中的新 API
