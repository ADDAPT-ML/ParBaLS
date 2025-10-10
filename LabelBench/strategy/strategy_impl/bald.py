from tqdm import tqdm
import numpy as np
import torch

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


class BALD(Strategy):
    strategy_name = "bald"

    def __init__(self, strategy_config, dataset):
        super(BALD, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_EMBEDDING]
        self.posterior_subsample = strategy_config["posterior_subsample"] if "posterior_subsample" in strategy_config else None
        self.batching = strategy_config["batching"] if "batching" in strategy_config else "topk"
        assert self.batching in ["topk", "softmax", "power", "softrank", "batch"]

    def select(self, trainer, budget, model):
        train_X = torch.from_numpy(trainer.retrieve_inputs(self.input_types)[0])
        unlabeled = self.dataset.unlabeled_idxs()
        pool_mc = model.predict_mc(train_X[unlabeled], T=self.posterior_subsample) # (T, N_pool, C)
        H_mc = -torch.sum(pool_mc * torch.log2(torch.clamp(pool_mc, 1e-12)), dim=2).mean(dim=0) # (N_pool,)
        T, N_pool, C = pool_mc.shape
        if self.batching == "batch":
            max_combo = 60000
            top_idxs = []
            candidate_pool_idxs = list(range(N_pool))
            p_joint_cond_t = torch.ones((T, 1)) # stores p(y_1,...,y_n | ω_t) for the currently selected batch
            for _ in tqdm(range(budget), desc="BatchBALD"):
                scores = []
                for candidate_pool_idx in candidate_pool_idxs:
                    candidate_preds = pool_mc[:, candidate_pool_idx, :] # (T, C)
                    # p(y_1,...,y_n, y_cand | ω_t) = p(y_1,...,y_n | ω_t) * p(y_cand | ω_t)
                    p_new_joint_cond_t = p_joint_cond_t.unsqueeze(2) * candidate_preds.unsqueeze(1) # (T, C^n, C)
                    p_new_joint_cond_t = p_new_joint_cond_t.reshape(T, -1) # (T, C^(n+1))
                    p_new_joint = torch.mean(p_new_joint_cond_t, dim=0) # p(y_1,...,y_n, y_cand) of shape (C^(n+1),)
                    H_joint = -(p_new_joint * torch.log2(torch.clamp(p_new_joint, 1e-12))).sum() # H(Y_batch U Y_candidate)
                    E_H_cond = H_mc[top_idxs].sum() + H_mc[candidate_pool_idx] # E[H(Y_batch U Y_candidate | ω)]
                    score = H_joint - E_H_cond # I(Y_batch U Y_cand; ω)
                    scores.append(score)
                best_candidate_local_idx = np.argmax(scores)
                best_candidate_pool_idx = candidate_pool_idxs.pop(best_candidate_local_idx)
                top_idxs.append(best_candidate_pool_idx)
                best_candidate_preds = pool_mc[:, best_candidate_pool_idx, :]
                p_joint_cond_t = (p_joint_cond_t.unsqueeze(2) * best_candidate_preds.unsqueeze(1)).reshape(T, -1) # Shape starts at (T, 1) and grows to (T, C), (T, C^2), ..., (T, C^n)
                if p_joint_cond_t.shape[1] > max_combo:
                    print(f"Reducing {p_joint_cond_t.shape[1]} combinations to {max_combo}...")
                    # p_joint_cond_t = p_joint_cond_t[:, :max_combo]
                    sampled_indices = torch.multinomial(p_joint_cond_t, num_samples=max_combo, replacement=True)
                    p_joint_cond_t = torch.gather(p_joint_cond_t, 1, sampled_indices)
                    p_joint_cond_t = p_joint_cond_t / p_joint_cond_t.sum(dim=1, keepdim=True)
        else:
            Pr_i = pool_mc.mean(dim=0) # (N_pool, C)
            H_i = -(Pr_i * torch.log2(torch.clamp(Pr_i, 1e-12))).sum(dim=1) # (N_pool,)
            scores = (H_i - H_mc).cpu().numpy()
            if self.batching == "softmax":
                top_idxs = self.get_softmax_samples(scores, beta=1, aquisition_batch_size=budget)
            elif self.batching == "power":
                top_idxs = self.get_power_samples(scores, beta=1, aquisition_batch_size=budget)
            elif self.batching == "softrank":
                top_idxs = self.get_softrank_samples(scores, beta=1, aquisition_batch_size=budget)
            else:
                top_idxs = np.argsort(scores)[-budget:]
        return unlabeled[top_idxs]
