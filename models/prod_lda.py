import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pytorch_lightning as pl


class ProdLDA(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 topic_size,
                 beta=2.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.vocab_size, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=100, out_features=self.topic_size*2),
        )
        self.encoder_norm_mu = nn.BatchNorm1d(num_features=self.topic_size, eps=0.001, momentum=0.001, affine=True)
        self.encoder_norm_mu.weight.data.copy_(torch.ones(self.topic_size))
        self.encoder_norm_mu.weight.requires_grad = False

        self.encoder_norm_logvar = nn.BatchNorm1d(num_features=self.topic_size, eps=0.001, momentum=0.001, affine=True)
        self.encoder_norm_logvar.weight.data.copy_(torch.ones(self.topic_size))
        self.encoder_norm_logvar.weight.requires_grad = False

        # decoder
        self.decoder = nn.Linear(in_features=self.topic_size, out_features=self.vocab_size)
        self.decoder_norm = nn.BatchNorm1d(num_features=self.vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.vocab_size))
        self.decoder_norm.weight.requires_grad = False

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        enc_out = F.softplus(self.encoder(x))
        mu = self.encoder_norm_mu(enc_out[:, :self.topic_size])
        logvar = self.encoder_norm_logvar(enc_out[:, self.topic_size:])
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        x_recon = F.log_softmax(self.decoder_norm(self.decoder(z)), dim=1)
        return x_recon, dist

    def training_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + kl
        self.log_dict({'train/loss': loss,
                       'train/recon': recon,
                       'train/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + kl
        self.log_dict({'val/loss': loss,
                       'val/recon': recon,
                       'val/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def objective(self, x, x_recon, dist):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        prior = Normal(torch.zeros(self.topic_size, device=x.device), torch.ones(self.topic_size, device=x.device))
        kl = self.beta * torch.distributions.kl.kl_divergence(dist, prior).mean()
        return recon, kl

    def get_topics(self, vocab, path):
        # load best model
        model = self.load_from_checkpoint(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get topics
        topics = model.decoder.weight.detach().cpu().numpy().T
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]
        return topics
