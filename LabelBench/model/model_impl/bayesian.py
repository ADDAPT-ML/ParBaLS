import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import pymc as pm
import arviz as az

from LabelBench.skeleton.model_skeleton import register_model


class Bayesian(nn.Module):

    def __init__(self, num_input, num_output, ret_emb, prior_sigma, sampling, draws, tune, chains):
        super(Bayesian, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.ret_emb = ret_emb
        self.prior_sigma = prior_sigma
        self.sampling = sampling
        assert self.sampling in ["nuts"]
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.trace = None

    def forward(self, features, ret_features=False, freeze=False):
        if freeze:
            features = features.data
        if ret_features:
            return self.predict(features), features.data
        elif self.ret_emb:
            return self.predict(features), features
        else:
            return self.predict(features)
    
    def build_pymc_model(self, X: np.ndarray, y: np.ndarray):
        model = pm.Model()
        with model:
            w = pm.Normal("w", mu=0, sigma=self.prior_sigma, shape=(self.num_output, self.num_input))
            b = pm.Normal("b", mu=0, sigma=self.prior_sigma, shape=(self.num_output,))
            logits = X @ w.T + b
            pm.Categorical("y", logit_p=logits, observed=y)
        return model

    def compute_hessian(self, map_dict, X: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian over a dataset with MAP."""
        W = torch.from_numpy(map_dict["w"].reshape(self.num_output, self.num_input)).to(X) # (C, D)
        B = torch.from_numpy(map_dict["b"].reshape(self.num_output)).to(X) # (C,)

        # Initialize Hessian blocks
        weight_dim = self.num_input * (self.num_output - 1)
        hessian_size = weight_dim + (self.num_output - 1)
        hessian = torch.zeros((hessian_size, hessian_size)).to(X)

        logits = torch.einsum("cd,nd->cn", W, X) + B[:, None] # (C, N)
        probs = torch.softmax(logits, dim=0) # (C, N)
        pi = probs.transpose(0, 1)[:, :-1] # (N, C-1)

        # Compute class uncertainty matrix M using torch.diag_embed for batch diagonal
        M = torch.diag_embed(pi) - torch.einsum('bi,bj->bij', pi, pi) # (N, C-1, C-1)
        xxT = torch.einsum('bi,bj->bij', X, X) # (N, D, D)

        # Compute Kronecker products for weight block
        kron_products = torch.einsum('bij,bkl->bikjl', M, xxT).reshape(X.shape[0], weight_dim, weight_dim)
        hessian[:weight_dim, :weight_dim] += kron_products.sum(dim=0)

        # Compute cross terms
        weight_bias_terms = torch.einsum('bij,bk->bikj', M, X).reshape(X.shape[0], weight_dim, self.num_output - 1)
        summed_weight_bias = weight_bias_terms.sum(dim=0)
        hessian[:weight_dim, weight_dim:] += summed_weight_bias
        hessian[weight_dim:, :weight_dim] += summed_weight_bias.T

        # Bias-bias block
        hessian[weight_dim:, weight_dim:] += M.sum(dim=0)

        return hessian

    def fit(self,
            features: torch.Tensor,
            labels: torch.Tensor):
        """
        Run MCMC on (features, labels) and store the trace.
        """
        X = features.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        if y.ndim == 2:
            y = y.argmax(axis=1)
        y = y.astype("int32")

        model = self.build_pymc_model(X, y)
        with model:
            self.trace = pm.sample(
                draws=self.draws, tune=self.tune,
                chains=self.chains, progressbar=True
            )
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute mean logits under the posterior:
          E_{p(w,b|D)}[ X @ w.T + b ].
        """
        if self.trace is None:
            print("Using prior for prediction without .fit()")
            return self.predict_prior(features).to(features.device)

        mc_probs = self.predict_mc(features, T=self.chains * self.draws) # (S, N, C)
        return torch.mean(mc_probs, dim=0) # (N, C)

    def predict_mc(self, X: torch.Tensor, T: int = None) -> torch.Tensor:
        """
        Return Monte-Carlo draws of predictive probabilities:
          - features: Tensor (N, D)
          - returns: np.array shape (T, N, C)
        If T is None, use all posterior samples; otherwise uniformly sub-sample T of them.
        """
        if self.trace is None:
            print("Using prior for prediction without .fit()")
            return self.predict_prior(X, num_samples=T)

        device = X.device

        w_samps = torch.from_numpy(self.trace.posterior["w"].stack(s=("chain","draw")).values).to(X) # (C, D, S)
        b_samps = torch.from_numpy(self.trace.posterior["b"].stack(s=("chain","draw")).values).to(X) # (C, S)

        S = w_samps.shape[2]

        if T is None or T >= S:
            idxs = torch.arange(S, device=device)
        else:
            idxs = torch.linspace(0, S - 1, T, dtype=torch.long, device=device)
        
        T = len(idxs)

        W = w_samps[:, :, idxs] # (C, D, T)
        B = b_samps[:, idxs] # (C, T)

        logits = torch.einsum("cdt,nd->cnt", W, X) + B.unsqueeze(1) # (C, N, T)
        probs = torch.softmax(logits, dim=0) # (C, N, T)
        return probs.permute(2, 1, 0) # (T, N, C)

    def predict_prior(self, features: torch.Tensor, num_samples: int = 1, W=None, B=None):
        X = features # (N, D)
        if W is None or B is None:
            # Sample random weights and bias from prior
            W = torch.normal(mean=0, std=self.prior_sigma, size=(self.num_output, self.num_input, num_samples)).to(X) # (C, D, T)
            B = torch.normal(mean=0, std=self.prior_sigma, size=(self.num_output, num_samples)).to(X) # (C, T)
        logits = torch.einsum("cdt,nd->cnt", W, X) + B.unsqueeze(1) # (C, N, T)
        probs = softmax(logits, dim=0).permute(2, 1, 0).squeeze() # (T, N, C) or (N, C)
        return probs

@register_model("bayesian")
def init_Bayesian(model_config):
    return Bayesian(model_config["input_dim"], model_config["num_output"],
                    ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                    prior_sigma=model_config["prior_sigma"], sampling=model_config["sampling"],
                    draws=model_config["draws"], tune=model_config["tune"], chains=model_config["chains"])
