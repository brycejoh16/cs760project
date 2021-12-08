from torch import  nn
import torch
import matplotlib.pyplot as plt
import labellingFunctions as lf


# need to have a manual seed for reproducibility purposes...
# but this could be in the wrong place.
torch.manual_seed(767)
nz = 128  # latent vector size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### generator adopted and trained from Sala , Et. al.
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

def torch2numpy(a:torch.Tensor):
    return a.detach().numpy()

def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)

def z():
    return create_noise(1,nz)
    
def load_generator():
    model = Generator(nz).to(device)
    model.load_state_dict(torch.load("generator_200_01.pth"))
    model.eval()
    return model

def try_to_load_generator():
    #  is equal to 128
    z=create_noise(1,nz)
    model = Generator(nz).to(device)
    model.load_state_dict(torch.load("generator01_test.pth"))
    model.eval()
    # we don't need the gradient anymore.
    with torch.no_grad():
        pz=model(z).view(28,28)
        plt.imshow(pz)
        plt.show()

if __name__ == '__main__':
    pass
    # for _  in range(8):
    #     try_to_load_generator()
