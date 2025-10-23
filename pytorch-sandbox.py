import torch
tensor1 = torch.tensor([[1,2],[2,4]]).to(torch.float32)
tensor2 = torch.tensor([[2,3],[5,7]]).to(torch.float32)
print(tensor1 @ tensor2.T)

n = int(input("回数を入力してください:"))
tensor1n = torch.linalg.matrix_power(tensor1, n)
print(tensor1n)