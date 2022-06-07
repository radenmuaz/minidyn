import torch


class A(torch.nn.Module):
    count = 0
    def __init__(self):
        super().__init__()
        A.count += 1
        self.id = self.count

class B(A):
    def __init__(self):
        super().__init__()
b=torch.jit.script(B())
b2=torch.jit.script(B())

print(b2.id)
