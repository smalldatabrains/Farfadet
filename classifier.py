import torch

class Classifier():
    def __init__(self):
        self.x = torch.rand(100, 1, requires_grad=True)
        self.noise = torch.rand(100,1)*0.1
        self.targets = 2*self.x+1+self.noise

    def add_ten(self, tensor):
        return tensor + 10

    def divide_two(self, tensor):
        return tensor * 0.5

    def polynome(self, tensor,noise):
        return 2 * tensor + 1 + noise

    def printTensor(self, tensor):
        print(tensor)
        print(self.x.grad)

    def gradient(self, tensor):
        tensor.backward(torch.ones_like(tensor))
    
    def MSELoss(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

if __name__ == "__main__":
    y = Classifier()
    z = y.add_ten(y.x)
    z = y.divide_two(z)
    p = y.polynome(z, y.noise)
    
    loss = y.MSELoss(p,y.targets)
    y.gradient(loss)
    y.printTensor(z)
    print("Loss:", loss.item())