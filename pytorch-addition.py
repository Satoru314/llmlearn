import torch
import os

# モデル定義（入力6、隠れ層128→64、出力1）軽量化！
class AddNet(torch.nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        self.fc1 = torch.nn.Linear(6, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# デバイス設定
device = torch.device("cpu")

# モデル・損失・最適化
model = AddNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 重みファイル
weight_path = "addnet_weights.pth"

# 既存の重みロード
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    print("✅ Saved weights loaded!")
else:
    print("ℹ️ No saved weights found. Starting fresh.")

# ワンホット演算子コード（+,-,*,/）
op2onehot = {
    "+": [1, 0, 0, 0],
    "-": [0, 1, 0, 0],
    "*": [0, 0, 1, 0],
    "/": [0, 0, 0, 1],
}
# 生成する整数の範囲（必要に応じて変更）
min_v, max_v = 1, 99

# 学習ループ
for epoch in range(50000):  # 回数も削減（軽量化したので早く収束するはず）
    # 各演算子を32サンプルずつ、計128サンプル
    a_list, b_list, y_list, op_list = [], [], [], []

    for op in ["+", "-", "*", "/"]:
        # 1以上の整数サンプリング（[min_v, max_v]）
        a_batch = torch.randint(min_v, max_v + 1, (32, 1), device=device, dtype=torch.int32).to(torch.float32)
        b_batch = torch.randint(min_v, max_v + 1, (32, 1), device=device, dtype=torch.int32).to(torch.float32)

        if op == "+":
            y_batch = a_batch + b_batch
        elif op == "-":
            y_batch = a_batch - b_batch
        elif op == "*":
            y_batch = a_batch * b_batch
        else:
            y_batch = a_batch / b_batch

        a_list.append(a_batch)
        b_list.append(b_batch)
        y_list.append(y_batch)
        op_list.extend([op] * 32)

    a = torch.cat(a_list)  # (128, 1)
    b = torch.cat(b_list)  # (128, 1)
    y = torch.cat(y_list)  # (128, 1)

    # 正規化！入力だけを0〜1の範囲に
    a_norm = a / 100.0  # 1〜99 → 0.01〜0.99
    b_norm = b / 100.0
    # 出力は正規化しない（そのまま使う）→ スケールの不均衡を防ぐ

    # ワンホットベクトル作成
    op_onehot_batch = torch.tensor([op2onehot[op] for op in op_list], dtype=torch.float32, device=device)
    # 入力を(128,6)に: [a_norm, b_norm, 1000/0100/0010/0001]
    x = torch.cat([a_norm, b_norm, op_onehot_batch], dim=1)

    # 順伝播（生の出力と比較）
    output = model(x)
    loss = criterion(output, y)

    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # たまに進捗表示
    if epoch % 2000 == 0:
        print(f"epoch {epoch:4d} | loss = {loss.item():.6f}")

# 重み保存
torch.save(model.state_dict(), weight_path)
print("💾 Weights saved!")

# テストしてみる（正規化を適用）
tests_meta = [
    (2.0, 5.0, '+'),
    (7.0, 3.0, '-'),
    (3.0, 4.0, '*'),
    (8.0, 2.0, '/'),
    (1.5, 1.0, '/'),
    (70.0, 35.0, '-'),
    (33.0, 43.0, '*'),
    (81.0, 2.0, '/'),
    (11.0, 88.0, '/'),
]
# 入力を正規化してテスト用行列を作成
tests_mat = torch.tensor(
    [[a/100.0, b/100.0] + op2onehot[op] for (a, b, op) in tests_meta],
    dtype=torch.float32,
    device=device
)

for i, (a, b, op) in enumerate(tests_meta):
    x_test = tests_mat[i:i+1]  # (1,6)
    with torch.no_grad():
        pred = model(x_test).item()  # 逆正規化不要（生の値で出力）
    if op == '+':
        true = a + b
    elif op == '-':
        true = a - b
    elif op == '*':
        true = a * b
    else:
        true = a / b
    print(f"{a} {op} {b} ≒ {pred:.3f} (true {true:.3f})")
