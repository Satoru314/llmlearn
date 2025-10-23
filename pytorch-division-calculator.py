import torch
import math

# モデル定義（学習時と同じ構造）
class DivNet(torch.nn.Module):
    def __init__(self):
        super(DivNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# デバイス設定
device = torch.device("cpu")

# モデルの読み込み
model = DivNet().to(device)
model.load_state_dict(torch.load("division_weights.pth", map_location=device, weights_only=True))
model.eval()  # 推論モードに設定

print("✅ Division model loaded successfully!")

def calculate(a, b):
    """AIで割り算を実行"""
    # 入力を正規化（学習時と同じ）
    a_norm = a / 100.0
    b_norm = b / 100.0

    # 入力テンソル作成
    x = torch.tensor([[a_norm, b_norm]], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_log = model(x).item()  # 対数スケールで出力
        result = math.exp(pred_log)  # expで元に戻す

    return result

# インタラクティブモード
print("\n➗ AI Division Calculator")
print("使い方: 10 / 2 のように入力してください")
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
            print("❌ 形式が正しくありません。例: 10 / 2")
            continue

        a_str, op, b_str = parts

        if op != '/':
            print("❌ 割り算専用です。'/' を使ってください")
            continue

        a = float(a_str)
        b = float(b_str)

        if b == 0:
            print("❌ ゼロ除算はできません")
            continue

        # AI計算
        ai_result = calculate(a, b)

        # 正解計算
        true_result = a / b

        # 結果表示
        error = abs(ai_result - true_result)
        error_pct = (error / true_result) * 100 if true_result != 0 else 0

        print(f"AI予測: {ai_result:.3f}")
        print(f"正解: {true_result:.3f}")
        print(f"誤差: {error:.3f} ({error_pct:.2f}%)")

    except ValueError:
        print("❌ 数値が正しくありません")
    except Exception as e:
        print(f"❌ エラー: {e}")
