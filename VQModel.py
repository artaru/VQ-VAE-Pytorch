import numpy as np

import torch
import torch.nn as nn

from Decoder import Decoder
from Encoder import Encoder
from VectorQuantizer import VectorQuantizer

class VQModel(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQModel, self).__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                       commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

if __name__ == "__main__":
    x = np.random.random_sample((1,3,400,600))
    x = torch.tensor(x).float()

    quant = Model(128, 2,132,512, 64, 0.25)
    print(quant(x)[1].shape)