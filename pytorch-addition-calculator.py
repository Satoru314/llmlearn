import torch

# モデル定義（学習時と同じ構造に合わせる）
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

# モデルの読み込み
model = AddNet().to(device)
model.load_state_dict(torch.load("addnet_weights.pth", map_location=device, weights_only=True))
model.eval()  # 推論モードに設定

print("✅ Model loaded successfully!")

# ワンホット演算子コード
op2onehot = {
    "+": [1, 0, 0, 0],
    "-": [0, 1, 0, 0],
    "*": [0, 0, 1, 0],
    "/": [0, 0, 0, 1],
}

def calculate(a, b, op):
    """AIで計算を実行"""
    # 入力を正規化（学習時と同じ）
    a_norm = a / 100.0
    b_norm = b / 100.0

    # 入力テンソル作成 [a_norm, b_norm, op_onehot]
    x = torch.tensor([[a_norm, b_norm] + op2onehot[op]], dtype=torch.float32, device=device)

    with torch.no_grad():
        result = model(x).item()  # 逆正規化不要（生の値で出力）

    return result

# インタラクティブモード
print("\n🧮 AI Calculator")
print("使い方: 2 + 5 のように入力してください")
print("終了するには 'q' を入力")

while True:
    try:
        user_input = input("\n計算式: ").strip()
        
        if user_input.lower() == 'q':
            print("終了します")
            break
        
        # 入力をパース
        parts = user_input.split()
        if len(parts) != 3:
            print("❌ 形式が正しくありません。例: 2 + 5")
            continue
        
        a_str, op, b_str = parts
        
        if op not in op2onehot:
            print(f"❌ 演算子 '{op}' は対応していません。使用可能: +, -, *, /")
            continue
        
        a = float(a_str)
        b = float(b_str)
        
        # AI計算
        ai_result = calculate(a, b, op)
        
        # 正解計算
        if op == '+':
            true_result = a + b
        elif op == '-':
            true_result = a - b
        elif op == '*':
            true_result = a * b
        else:
            true_result = a / b if b != 0 else float('inf')
        
        print(f"AI予測: {ai_result:.3f}")
        print(f"正解: {true_result:.3f}")
        print(f"誤差: {abs(ai_result - true_result):.3f}")
        
    except ValueError:
        print("❌ 数値が正しくありません")
    except Exception as e:
        print(f"❌ エラー: {e}")