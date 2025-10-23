import torch
import os

class MLP(torch.nn.Module):
    def __init__(self, input_size=100, hidden1=128, hidden2=64, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# モデルの作成
model = MLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 重みファイルのパス
weight_path = "mlp_weights.pth"

# 🔹 もし重みファイルがあれば読み込み
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    print("✅ Saved weights loaded!")
else:
    print("ℹ️ No saved weights found. Starting fresh.")

# ダミーデータ
x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))

# 学習ステップ
outputs = model(x)
loss = criterion(outputs, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# 🔹 更新後の重みを保存
torch.save(model.state_dict(), weight_path)
print("💾 Weights saved!")
print("loss:", loss.item())
