import torch
import torch.nn as nn
import os
import math

# モデル定義（入力2、隠れ層64→32、出力1）
class DivNet(nn.Module):
    def __init__(self):
        super(DivNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# デバイス設定
device = torch.device("cpu")

# モデル・損失・最適化
model = DivNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 重みファイル
weight_path = "division_weights.pth"

# 既存の重みロード
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    print("✅ Saved weights loaded!")
else:
    print("ℹ️ No saved weights found. Starting fresh.")

# 生成する整数の範囲
min_a, max_a = 1, 99    # 分子: 1〜99
min_b, max_b = 10, 99   # 分母: 10〜99（安定化のため10以上）

# 学習ループ
print("\n🎯 Training division model with log-scale...")
for epoch in range(10000):
    # 128サンプル生成
    a = torch.randint(min_a, max_a + 1, (128, 1), device=device, dtype=torch.int32).to(torch.float32)
    b = torch.randint(min_b, max_b + 1, (128, 1), device=device, dtype=torch.int32).to(torch.float32)

    # 入力正規化
    a_norm = a / 100.0  # 0.01〜0.99
    b_norm = b / 100.0  # 0.10〜0.99

    # 出力を対数変換（これが重要！）
    y = a / b  # 実際の割り算結果
    y_log = torch.log(y)  # 対数を取る

    # 入力テンソル作成
    x = torch.cat([a_norm, b_norm], dim=1)  # (128, 2)

    # 順伝播
    output = model(x)

    # 重み付き損失計算（分母が小さいほど重要視）
    weight = 100.0 / b  # b=10なら10倍、b=50なら2倍、b=100なら1倍
    squared_errors = (output - y_log) ** 2
    weighted_loss = (weight * squared_errors).mean()
    loss = weighted_loss

    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 進捗表示
    if epoch % 2000 == 0:
        print(f"epoch {epoch:4d} | loss = {loss.item():.6f}")

# 重み保存
torch.save(model.state_dict(), weight_path)
print("💾 Weights saved!")

# テスト（分母が小さいケースを多めに）
print("\n📊 Testing...")
tests = [
    (90.0, 10.0),   # 分母小 → 結果大
    (50.0, 10.0),   # 分母小 → 結果大
    (99.0, 11.0),   # 分母小 → 結果大
    (81.0, 15.0),   # 分母小 → 結果大
    (50.0, 25.0),   # 分母中 → 結果中
    (81.0, 27.0),   # 分母中 → 結果中
    (15.0, 30.0),   # 分母大 → 結果小
    (45.0, 90.0),   # 分母大 → 結果小
]

for a, b in tests:
    # 正規化
    a_norm = a / 100.0
    b_norm = b / 100.0

    x_test = torch.tensor([[a_norm, b_norm]], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_log = model(x_test).item()
        pred = math.exp(pred_log)  # 対数から戻す

    true = a / b
    error = abs(pred - true)
    error_pct = (error / true) * 100 if true != 0 else 0

    print(f"{a:5.1f} / {b:5.1f} = {pred:7.3f} (true: {true:7.3f}, error: {error:6.3f}, {error_pct:5.2f}%)")

print("\n✅ Training complete!")
