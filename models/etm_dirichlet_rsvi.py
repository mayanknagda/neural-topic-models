import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl


def calc_epsilon(p, alpha):
    sqrt_alpha = torch.sqrt(9 * alpha - 3)
    powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
    return sqrt_alpha * (powza - 1)


def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = u.shape[0]
    K = alpha.shape[1]
    r = torch.arange(B, device=alpha.device)
    rm = torch.reshape(r, (-1, 1, 1)).float()
    alpha_vec = torch.tile(alpha, (B, 1)).reshape((B, -1, K)) + rm
    u_pow = torch.pow(u, 1. / alpha_vec) + 1e-10
    return torch.prod(u_pow, axis=0) * gamma_h(epsilon, alpha + B)


def gamma_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return b * (v ** 3)


def rsvi(alpha):
    B = 10
    gam = torch.distributions.Gamma(alpha + B, 1).sample().to(alpha.device)
    eps = calc_epsilon(gam, alpha + B).detach().to(alpha.device)
    u = torch.rand((B, alpha.shape[0], alpha.shape[1]), device=alpha.device)
    doc_vec = gamma_h_boosted(eps, u, alpha)
    # normalize
    gam = doc_vec
    doc_vec = gam / torch.reshape(torch.sum(gam, dim=1), (-1, 1))
    z = doc_vec.reshape(alpha.shape)
    return z


class ETMD_RSVI(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 vocab_embeddings,
                 topic_size,
                 beta=2.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.vocab_embeddings = vocab_embeddings
        self.topic_size = topic_size
        self.beta = beta
        self.embedding_size = vocab_embeddings.shape[1]

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.vocab_size, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=100, out_features=self.topic_size),
        )
        self.encoder_norm = nn.BatchNorm1d(num_features=self.topic_size, eps=0.001, momentum=0.001, affine=True)
        self.encoder_norm.weight.data.copy_(torch.ones(self.topic_size))
        self.encoder_norm.weight.requires_grad = False

        # decoder
        self.topic_embeddings = nn.Linear(self.topic_size, self.embedding_size, bias=False)
        self.word_embeddings = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        # initialize linear layer with pre-trained embeddings
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        self.decoder_norm = nn.BatchNorm1d(num_features=self.vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.vocab_size))
        self.decoder_norm.weight.requires_grad = False

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        z = rsvi(alpha)
        topic_embeddings = self.topic_embeddings(z)  # (batch_size, 300)
        word_embeddings = self.word_embeddings.weight  # (vocab_size, 300)
        # dot product
        x_recon = torch.matmul(topic_embeddings, word_embeddings.T)  # (batch_size, vocab_size)
        x_recon = F.log_softmax(self.decoder_norm(x_recon), dim=1)  # (batch_size, vocab_size)
        return x_recon, alpha

    def training_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, alpha = self(x)
        recon, kl = self.objective(x, x_recon, alpha)
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
        x_recon, alpha = self(x)
        recon, kl = self.objective(x, x_recon, alpha)
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

    def objective(self, x, x_recon, alpha):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        kl = self.kl_divergence(alpha).mean()
        return recon, kl

    def kl_divergence(self, alpha):
        alpha_prior = torch.ones(alpha.shape, device=alpha.device) * 0.02
        prior = Dirichlet(alpha_prior)
        posterior = Dirichlet(alpha)
        return self.beta * torch.distributions.kl.kl_divergence(posterior, prior)

    def get_topics(self, vocab, path):
        # load best model
        model = self.load_from_checkpoint(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get topics
        topic_embeddings = model.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings = model.word_embeddings.weight.data.cpu().numpy().T  # (E, V)
        topics = topic_embeddings @ word_embeddings  # (K, V)
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]
        return topics