import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributions import Dirichlet
import pytorch_lightning as pl


def calc_epsilon(p, alpha):
    sqrt_alpha = torch.sqrt(9 * alpha - 3)
    powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
    return sqrt_alpha * (powza - 1)


def gamma_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return b * (v ** 3)


def gamma_grad_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return v ** 3 - 13.5 * eps * b * (v ** 2) * (c ** 3)


class RSVI(Function):
    @staticmethod
    def forward(ctx, alpha):
        p = torch.distributions.Gamma(alpha, 1).sample()
        eps = calc_epsilon(p, alpha)
        ctx.save_for_backward(alpha, eps)
        z = gamma_h(eps, alpha)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        alpha, eps = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input += gamma_grad_h(eps, alpha)
        return grad_input


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class DVAE_RSVI(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 topic_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.topic_size = topic_size

        # dropout
        self.dropout = nn.Dropout(p=0.25)

        # encoder
        self.weight_one = nn.Parameter(torch.empty(100, self.vocab_size))
        self.bias_one = nn.Parameter(torch.empty(100))

        nn.init.xavier_uniform_(self.weight_one)
        nn.init.uniform_(self.bias_one)

        self.weight_two = nn.Parameter(torch.empty(self.topic_size, 100))
        self.bias_two = nn.Parameter(torch.empty(self.topic_size))

        nn.init.xavier_uniform_(self.weight_two)
        nn.init.uniform_(self.bias_two)

        self.encoder_norm = nn.BatchNorm1d(num_features=self.topic_size, eps=0.001, momentum=0.001, affine=True)
        self.encoder_norm.weight.data.copy_(torch.ones(self.topic_size))
        self.encoder_norm.weight.requires_grad = False

        # decoder
        self.decoder_weight = nn.Parameter(torch.empty(self.vocab_size, self.topic_size))
        self.decoder_bias = nn.Parameter(torch.empty(self.vocab_size))

        nn.init.xavier_uniform_(self.decoder_weight)
        nn.init.uniform_(self.decoder_bias)

        self.decoder_norm = nn.BatchNorm1d(num_features=self.vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.vocab_size))
        self.decoder_norm.weight.requires_grad = False

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        lambda_ = F.relu(LinearFunction.apply(x, self.weight_one, self.bias_one))
        lambda_ = self.dropout(lambda_)
        l_1 = self.encoder_norm(LinearFunction.apply(lambda_, self.weight_two, self.bias_two))
        alpha = torch.max(torch.tensor(0.00001, device=x.device), F.softplus(l_1))
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        l_2 = LinearFunction.apply(z, self.decoder_weight, self.decoder_bias)
        l_2 = self.decoder_norm(l_2)
        x_recon = F.log_softmax(l_2, dim=1)
        return x_recon, alpha

    def training_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, alpha = self(x)
        recon, kl = self.objective(x, x_recon, alpha)
        loss = recon + 2 * kl
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
        alpha_prior = torch.ones((x.shape[0], self.topic_size), device=x.device) * 0.02
        kl = self.kl_divergence(alpha, alpha_prior).mean()
        return recon, kl

    def kl_divergence(self, alpha, alpha_prior):
        first_term = torch.lgamma(torch.sum(alpha, dim=1))
        second_term = torch.lgamma(torch.sum(alpha_prior, dim=1))
        third_term = torch.sum(torch.lgamma(alpha_prior), dim=1)
        fourth_term = torch.sum(torch.lgamma(alpha), dim=1)
        minus = alpha - alpha_prior
        digamma = torch.digamma(alpha) - torch.digamma(torch.sum(alpha, dim=1)).unsqueeze(1)
        fifth_term = torch.sum(minus * digamma, dim=1)
        return first_term - second_term + third_term - fourth_term + fifth_term

    def get_topics(self, vocab, path):
        # load best model
        model = self.load_from_checkpoint(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get topics
        topics = model.decoder_weight.detach().cpu().numpy().T
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]
        return topics
