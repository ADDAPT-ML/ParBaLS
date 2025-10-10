import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LabelBench.skeleton.trainer_skeleton import Trainer
from LabelBench.trainer.utils import EarlyStopping


class PyTorchPassiveTrainer(Trainer):
    trainer_name = "pytorch_passive"

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, feature_extractor):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, feature_extractor)

    def init_train(self, finetune_model):
        if finetune_model is None:
            model = self.model_fn(self.model_config)
            if "template" in self.model_config:
                if hasattr(model, 'init_head_withzeroshot'):
                    model.init_head_withzeroshot(classnames=self.dataset.get_classnames(),
                                                 template=self.model_config["template"])
                else:
                    raise ValueError("Please use a model that supports zero-shot learning.")
            model = model.to(self.device)
        else:
            model = copy.deepcopy(finetune_model).to(self.device)

        loss_fn = self.trainer_config["loss_fn"]
        max_epoch = self.trainer_config["max_epoch"]

        if self.device == "cuda":
            devices = list(range(torch.cuda.device_count()))
            print('Using devices', devices)
            model = torch.nn.parallel.DataParallel(model, device_ids=devices)

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = self.trainer_config["optim_fn"](params)
        total_steps = self.trainer_config["max_epoch"] * \
                      len(self.dataset.labeled_idxs()) // self.trainer_config["train_batch_size"]

        if "scheduler_fn" in self.trainer_config:
            scheduler = self.trainer_config["scheduler_fn"](optimizer, total_steps)
        else:
            scheduler = None

        # Initialize the early_stopping object.
        if "early_stop" in self.trainer_config and self.trainer_config["early_stop"]:
            early_stopping = EarlyStopping(
                patience=self.trainer_config["patience"] if "patience" in self.trainer_config else 5, verbose=True)
        else:
            early_stopping = None

        # Check to avoid using customized transform for embedding dataset.
        if "use_customized_transform" in self.model_config and "use_embeddings" in self.trainer_config:
            assert not (self.trainer_config["use_embeddings"] and self.model_config["use_customized_transform"]), \
                "Customized transform is only supported for non-embedding models."
        return model, params, loss_fn, max_epoch, optimizer, scheduler, early_stopping

    def check_early_stop(self, early_stopping, model):
        if early_stopping is not None:
            # Early_stopping needs the validation loss to check if it has decreased. If it has, it will make a
            # checkpoint of the current model.
            _, _, valid_losses, _ = self._test("val", model, **self.trainer_config)
            early_stopping(valid_losses.mean(), model=model)
            if early_stopping.early_stop:
                print("Early stopping.")
                return True
        return False

    @staticmethod
    def scheduler_step(scheduler, counter):
        if scheduler is not None:
            try:
                scheduler(counter)
            except:
                scheduler.step(counter)
            counter += 1
        return counter

    def train(self, finetune_model=None, finetune_config=None, return_input=False):
        if self.model_config["model_name"] not in ["bayesian"]:
            model, params, loss_fn, max_epoch, optimizer, scheduler, early_stopping = self.init_train(finetune_model)
        else:
            model = self.model_fn(self.model_config)
            max_epoch = 1

        # Get the training dataset for the non-embedding dataset.
        if "use_embeddings" not in self.trainer_config or (not self.trainer_config["use_embeddings"]):
            train_dataset, _, _ = self.dataset.get_input_datasets()
            if "use_customized_transform" in self.model_config and self.model_config["use_customized_transform"]:
                transform = model.module.get_preprocess(split="train")
                train_dataset.set_transform(transform)

        mixup_fn = self.trainer_config["mixup_fn"](self.dataset.num_classes)

        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
            # For each epoch, update the embedding dataset and use the saved embedding dataset epoch.
            if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
                self.dataset.update_embedding_dataset(epoch=epoch, feature_extractor=self.feature_extractor)
                train_dataset, _, _ = self.dataset.get_embedding_datasets()

            class_weights = 1. / np.clip(np.sum(self.dataset.get_train_labels(), axis=0), a_min=1, a_max=None)
            class_weights = torch.from_numpy(class_weights).float().to(self.device)

            if epoch == 0 or ("use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]):
                # Only use labeled examples for training.
                cur_train_dataset = Subset(train_dataset, self.dataset.labeled_idxs())
                loader = DataLoader(cur_train_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=True,
                                    num_workers=self.trainer_config["num_workers"],
                                    drop_last=(len(train_dataset) >= self.trainer_config["train_batch_size"]) and (1.5 * self.trainer_config["budget_per_iter"] >= self.trainer_config["train_batch_size"]))

            losses = 0
            total = 0

            if self.model_config["model_name"] in ["bayesian"]:
                features = []
                labels = []

            for img, target, *other in tqdm(loader, desc="Batch Index"):
                img, target = img.float().to(self.device), target.float().to(self.device)
                if mixup_fn is not None:
                    # Since the target is one-hot and the mixup function accepts class index only, we need to convert it
                    # to the class index.
                    target = torch.argmax(target, dim=1)
                    img, target = mixup_fn(img, target)

                if self.model_config["model_name"] in ["bayesian"]:
                    features.append(img.cpu())
                    labels.append(target.cpu())
                else:
                    with torch.amp.autocast(self.device):
                        if "init_freeze" in self.trainer_config and self.trainer_config["init_freeze"] > epoch:
                            if self.model_config["ret_emb"]:
                                pred, _ = model(img, ret_features=True, freeze=True)
                                pred = pred.squeeze(-1)
                            else:
                                pred = model(img, ret_features=False, freeze=True).squeeze(-1)
                        else:
                            if self.model_config["ret_emb"]:
                                pred, _ = model(img, ret_features=True)
                                pred = pred.squeeze(-1)
                            else:
                                pred = model(img, ret_features=False).squeeze(-1)

                        if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
                            loss = loss_fn(pred, target, weight=class_weights)
                        else:
                            loss = loss_fn(pred, target)

                    
                    counter = self.scheduler_step(scheduler, counter)
                    optimizer.zero_grad()
                    loss.backward()
                    if "clip_grad" in self.trainer_config:
                        nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                    optimizer.step()
                    losses += loss.detach().cpu().numpy()
                    total += len(pred)

            if self.model_config["model_name"] in ["bayesian"]:
                if len(features) > 0 and len(labels) > 0:
                    features = torch.cat(features, dim=0)
                    labels = torch.cat(labels, dim=0)
                    model.fit(features, labels)
                if return_input:
                    return model, 0, features, labels
                return model, 0

            losses = losses / total if total else losses
            print(f'Epoch {epoch} train loss: {losses}')

            if self.check_early_stop(early_stopping, model):
                break

        if early_stopping is not None:
            model.load_state_dict(early_stopping.best_state_dict)
        return model, losses

    def _test(self, dataset_split, model, **kwargs):
        model.eval()
        self.dataset.eval()
        if "mc_dropout" in kwargs and kwargs["mc_dropout"]:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
            datasets = self.dataset.get_embedding_datasets()
            assert all(dataset is not None for dataset in datasets), "Embedding features not found."
        else:
            datasets = self.dataset.get_input_datasets()

        if dataset_split == "train":
            dataset = datasets[0]
        elif dataset_split == "val":
            dataset = datasets[1]
        elif dataset_split == "test":
            dataset = datasets[2]

        # If clip model, we need to update the transform of dataset.
        if "use_customized_transform" in self.model_config and self.model_config["use_customized_transform"]:
            transform = model.module.get_preprocess(split="test")
            dataset.set_transform(transform)

        loader = DataLoader(dataset, batch_size=self.trainer_config["test_batch_size"], shuffle=False, num_workers=0)
        preds = np.zeros((len(dataset), self.dataset.num_classes), dtype=float)
        labels = np.zeros((len(dataset), self.dataset.num_classes), dtype=float)
        losses = np.zeros(len(dataset), dtype=float)
        embs = None
        counter = 0
        for img, target in loader:
            img, target = img.float().to(self.device), target.float().to(self.device)
            with torch.no_grad(), torch.amp.autocast(self.device):
                pred, emb = model(img, ret_features=True)
                if embs is None:
                    embs = np.zeros((len(dataset), emb.shape[-1]), dtype=float)
                pred = pred.squeeze(-1)
                loss = self.trainer_config["loss_fn"](pred, target)
            preds[counter: (counter + len(pred))] = self.trainer_config["pred_fn"](pred).cpu().numpy()
            labels[counter: (counter + len(pred))] = target.cpu().numpy()
            losses[counter: (counter + len(pred))] = loss.cpu().numpy()
            embs[counter: (counter + len(pred))] = emb.cpu().numpy()
            counter += len(pred)
        assert counter == len(preds)
        model.train()
        self.dataset.train()
        return preds, labels, losses, embs
