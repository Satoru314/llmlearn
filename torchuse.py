import torch
print(torch.__version__)
print(torch.cuda.is_available())

# デバイスの自動選択（推奨）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# テンソルの作成例
x = torch.randn(3, 3).to(device)
print(x)