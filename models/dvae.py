import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import pytorch_lightning as pl


class DVAE(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 topic_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.topic_size = topic_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=self.topic_size),
            nn.BatchNorm1d(num_features=self.topic_size, affine=False),
            nn.Softplus(),
        )

        # decoder
        self.decoder = nn.Linear(in_features=self.topic_size, out_features=self.vocab_size)
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        alpha = self.encoder(x)
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        x_recon = self.decoder_norm(self.decoder(z))
        return x_recon, dist

    def training_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + 2 * kl
        self.log_dict({'train/loss': loss, 'train/recon': recon, 'train/kl': kl}, prog_bar=True, logger=True,
                      on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + 2 * kl
        self.log_dict({'val/loss': loss, 'val/recon': recon, 'val/kl': kl}, prog_bar=True, logger=True, on_step=False,
                      on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def objective(self, x, x_recon, dist):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        prior = Dirichlet(torch.ones(self.topic_size, device=x.device) * 0.02)
        kl = torch.distributions.kl.kl_divergence(dist, prior).mean()
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
