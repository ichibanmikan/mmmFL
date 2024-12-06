import torch
from torch.amp import GradScaler, autocast

# 创建 GradScaler 和模型
scaler = GradScaler()
model = torch.nn.Linear(10, 1).cuda()

# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 定义一些训练数据和目标
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()

# 使用 GradScaler 进行自动混合精度训练
for i in range(1000):
    optimizer.zero_grad()

    # 将前向传递包装在autocast中以启用混合精度
    with autocast(device_type="cuda"):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

    # 调用 GradScaler 的 backward() 方法计算梯度并缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if i % 100 == 0:
        print(f"Step {i}, loss={loss.item():.4f}")
