import torch as T
import torch.nn as nn
from collections import OrderedDict

class VAEUnit(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 stride=2, padding=1, mode='encoder'):
        super().__init__()
        if mode == 'encoder':
            self.layer = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=4, stride=2, padding=2)
        elif mode == 'decoder':
            self.layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                            kernel_size=4, stride=stride, 
                                            padding=padding)
            
        self.batch_norm_layer = nn.BatchNorm2d(out_channels)
        self.activation_layer = nn.PReLU()
        
    def forward(self, x):
        z = self.layer(x)
        z = self.batch_norm_layer(z)
        z = self.activation_layer(z)
        return z
    

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(OrderedDict([
            ('Layer_start',nn.Conv2d(in_channels=1, out_channels=32, 
                                     kernel_size=4, stride=2, padding=2)), 
            ('Activation_layer', nn.PReLU()),
            ('ConvUnit_1', VAEUnit(32, 64, mode='encoder')),
            ('ConvUnit_2', VAEUnit(64, 128, mode='encoder')),
            ('ConvUnit_3', VAEUnit(128, 256, mode='encoder'))]))
        
        self.mean_layer = nn.Linear(3*3*256 ,32) 
        self.logvar_layer = nn.Linear(3*3*256, 32) 
        self.latent_layer = nn.Linear(32, 3*3*256) 
        
        self.decoder = nn.Sequential(OrderedDict([
            ('DeConvUnit_1', VAEUnit(256, 128, stride=2, 
                                     padding=2, mode='decoder')),
            ('DeConvUnit_2', VAEUnit(128, 64, stride=2, 
                                     padding=1, mode='decoder')),
            ('DeConvUnit_3', VAEUnit(64, 32, stride=2, 
                                     padding=2, mode='decoder')),
            ('Layer_end', nn.ConvTranspose2d(in_channels=32, out_channels=1, 
                                             kernel_size=4, stride=2, padding=1)),
            ('Activation_layer', nn.Sigmoid())]))
        
    def encode(self, x):
        z = self.encoder(x)
        z = T.reshape(z, (-1, 3*3*256))
        mean = self.mean_layer(z)
        logvar = self.logvar_layer(z)
        return mean, logvar
    
    def decode(self, x):
        z = self.latent_layer(x)
        z = T.reshape(z, (-1, 256, 3, 3))
        z = self.decoder(z)
        return 255*z

    def reparameterize_trick(self, mean, logvar):
        std_dev = T.exp(logvar / 2.0)
        epsilon = T.randn_like(std_dev)
        return epsilon * std_dev + mean
    
    def sample_latent_vector(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize_trick(mean, logvar)
        z = z.detach().cpu()
        return z.squeeze().numpy()
    
    def forward(self, x, encoder=False, decoder=True):
        mean, logvar = self.encode(x)
        z = self.reparameterize_trick(mean, logvar)
        if encoder:
            return z
        elif decoder:
            return self.decode(z), mean, logvar