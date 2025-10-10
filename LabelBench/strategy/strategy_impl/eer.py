import numpy as np
import torch

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


class EERSampling(Strategy):
    strategy_name = "eer"

    def __init__(self, strategy_config, dataset):
        super(EERSampling, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_EMBEDDING, ALInput.VAL_EMBEDDING]
        self.binary = True if "binary" in strategy_config and strategy_config["binary"] == True else False
        self.posterior_subsample = strategy_config["posterior_subsample"] if "posterior_subsample" in strategy_config else None
        self.pool_subsample = strategy_config["pool_subsample"] if "pool_subsample" in strategy_config else None
        self.val_subsample = strategy_config["val_subsample"] if "val_subsample" in strategy_config else None
        self.batching = strategy_config["batching"] if "batching" in strategy_config else "topk"
        assert self.batching in ["topk", "softmax", "power", "softrank", "single-pseudo", "multi-pseudo"]

    def select(self, trainer, budget, model, labeled_features=None, labeled_labels=None):
        train_X, val_X = trainer.retrieve_inputs(self.input_types)
        train_X = torch.from_numpy(train_X)
        val_X = torch.from_numpy(val_X)

        full_unlabeled = np.array(self.dataset.unlabeled_idxs(), dtype=int)
        if self.pool_subsample and len(full_unlabeled) > self.pool_subsample:
            pool_idxs = np.random.choice(full_unlabeled, self.pool_subsample, replace=False)
        else:
            pool_idxs = full_unlabeled

        full_val = np.arange(val_X.shape[0], dtype=int)
        if self.val_subsample and len(full_val) > self.val_subsample:
            val_idxs = np.random.choice(full_val, self.val_subsample, replace=False)
        else:
            val_idxs = full_val

        pool_mc = model.predict_mc(train_X[pool_idxs], T=self.posterior_subsample) # (T, N_pool, C)
        val_mc = model.predict_mc(val_X[val_idxs], T=self.posterior_subsample) # (T, N_val, C)

        T, N_pool, C = pool_mc.shape
        if self.binary and C != 2:
            self.binary = False
        N_val = val_mc.shape[1]

        if self.batching == "single-pseudo":
            top_idxs = []
            candidate_pool_idxs = list(range(N_pool))
            for _ in range(budget):
                Pr_joint = self.compute_Pr_joint(pool_mc, val_mc)
                scores = self.compute_scores(pool_mc, Pr_joint, N_val)
                best_candidate_local_idx = torch.argmax(scores).item()
                best_candidate_pool_idx = candidate_pool_idxs.pop(best_candidate_local_idx)
                top_idxs.append(best_candidate_pool_idx)
                pseudo_pool_idxs = pool_idxs[candidate_pool_idxs]
                pseudo_model = trainer.model_fn(trainer.model_config)
                labeled_features = torch.cat([labeled_features, torch.tensor(train_X[pool_idxs[best_candidate_pool_idx]]).reshape(1, -1)], dim=0)
                pseudo_label = torch.zeros((1, C))
                pseudo_label[0, np.argmax(pool_mc[:, best_candidate_local_idx, :].mean(axis=0))] = 1
                labeled_labels = torch.cat([labeled_labels, pseudo_label], dim=0)
                pseudo_model.fit(labeled_features, labeled_labels)
                pool_mc = pseudo_model.predict_mc(train_X[pseudo_pool_idxs], T=self.posterior_subsample) # (T, N_pool, C)
                val_mc = pseudo_model.predict_mc(val_X[val_idxs], T=self.posterior_subsample) # (T, N_val, C)
        elif self.batching == "multi-pseudo":
            pseudo_num = 10
            pseudo_label = np.argmax(pool_mc[np.random.choice(range(T), pseudo_num, replace=False)], axis=2) # (k', N_pool)
            pseudo_label = torch.tensor(np.eye(C)[pseudo_label]) # (k', N_pool, C)
            pool_mc = [pool_mc]
            val_mc = [val_mc]
            top_idxs = []
            candidate_pool_idxs = list(range(N_pool))
            for sample_idx in range(budget):
                scores = torch.zeros(N_pool - sample_idx)
                for pseudo_idx in range(len(pool_mc)):
                    Pr_joint = self.compute_Pr_joint(pool_mc[pseudo_idx], val_mc[pseudo_idx])
                    scores = self.compute_scores(pool_mc[pseudo_idx], Pr_joint, N_val, scores)
                best_candidate_local_idx = torch.argmax(scores).item()
                best_candidate_pool_idx = candidate_pool_idxs.pop(best_candidate_local_idx)
                top_idxs.append(best_candidate_pool_idx)
                pseudo_pool_idxs = pool_idxs[candidate_pool_idxs]
                pseudo_model = trainer.model_fn(trainer.model_config)
                labeled_features = torch.cat([labeled_features, torch.tensor(train_X[pool_idxs[best_candidate_pool_idx]]).reshape(1, -1)], dim=0)
                new_pool_mc = []
                new_val_mc = []
                for pseudo_idx in range(pseudo_num):
                    pseudo_model = trainer.model_fn(trainer.model_config)
                    semipseudo_labels = torch.cat([labeled_labels, pseudo_label[pseudo_idx, top_idxs]], dim=0)
                    pseudo_model.fit(labeled_features, semipseudo_labels)
                    new_pool_mc.append(pseudo_model.predict_mc(train_X[pseudo_pool_idxs], T=self.posterior_subsample)) # (T, N_pool, C)
                    new_val_mc.append(pseudo_model.predict_mc(val_X[val_idxs], T=self.posterior_subsample)) # (T, N_val, C)
                pool_mc = new_pool_mc
                val_mc = new_val_mc
        else:
            print("Computing Pr_joint...")
            Pr_joint = self.compute_Pr_joint(pool_mc, val_mc)
            scores = self.compute_scores(pool_mc, Pr_joint, N_val).cpu().numpy()

            if self.batching == "softmax":
                top_idxs = self.get_softmax_samples(scores, beta=1, aquisition_batch_size=budget)
            elif self.batching == "power":
                top_idxs = self.get_power_samples(scores, beta=1, aquisition_batch_size=budget)
            elif self.batching == "softrank":
                top_idxs = self.get_softrank_samples(scores, beta=1, aquisition_batch_size=budget)
            else:
                top_idxs = np.argsort(scores)[-budget:]
        return pool_idxs[top_idxs].tolist()

    def compute_Pr_joint(self, pool_mc, val_mc):
        if self.binary:
            Pr_pool_pos = torch.mean(pool_mc[:, :, 1], dim=0)
            Pr_val_pos = torch.mean(val_mc[:, :, 1], dim=0)
            Pr_joint_pos = torch.einsum("tp,tv->pv", pool_mc[:, :, 1], val_mc[:, :, 1]) / float(self.posterior_subsample) # (N_pool, N_val)
            p11 = Pr_joint_pos
            p10 = Pr_pool_pos.unsqueeze(1) - p11
            p01 = Pr_val_pos.unsqueeze(0) - p11
            p00 = 1 - p10 - p01 - p11
            Pr_joint = torch.stack([torch.stack([p00, p01], dim=-1), torch.stack([p10, p11], dim=-1)], dim=-2) # (N_pool, N_val, 2, 2)
        else:
            Pr_joint = torch.einsum("tpc,tvd->pvcd", pool_mc, val_mc) / float(self.posterior_subsample) # (N_pool, N_val, C, C)
        return Pr_joint
    
    def compute_scores(self, pool_mc, Pr_joint, N_val, scores=None):
        # Score_i = N_val * H(Y_i) - sum_j H(Y_j, Y_i)
        Pr_i = pool_mc.mean(dim=0) # (N_pool, C)
        H_i  = -(Pr_i * torch.log2(torch.clamp(Pr_i, 1e-12))).sum(dim=1) # (N_pool,)
        H_j_sum = - torch.sum(Pr_joint * torch.log2(torch.clamp(Pr_joint, 1e-12)), dim=(1, 2, 3)) # (N_pool,)
        if scores is not None:
            scores += N_val * H_i - H_j_sum
        else:
            scores = N_val * H_i - H_j_sum
        return scores
