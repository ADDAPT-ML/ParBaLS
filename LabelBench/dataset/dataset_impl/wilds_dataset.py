import torch
import torch.nn.functional as F
import numpy as np
import wilds
from torch.utils.data import Subset
from torchvision import transforms

from LabelBench.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType, TransformDataset
from LabelBench.dataset.dataset_impl.label_name.classnames import get_classnames


def get_wilds_datasets(data_dir, dataset_name):
    wilds_dataset = wilds.get_dataset(dataset=dataset_name, root_dir=data_dir, download=True)
    train_dataset = wilds_dataset.get_subset("train")
    val_dataset = wilds_dataset.get_subset("id_val")
    test_dataset = wilds_dataset.get_subset("id_test")
    print("Number of train, val, test points:")
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, val_dataset, test_dataset

def transform_wilds_datasets(n_class, dataset_name, train_dataset, val_dataset, test_dataset):
    full_classnames = get_classnames(dataset_name)
    full_n_class = len(full_classnames)
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])

    if n_class < full_n_class:
        classnames = full_classnames[:n_class] + ("others",)
    else:
        n_class = full_n_class
        classnames = full_classnames

    return TransformDataset(train_dataset, transform=train_transform, target_transform=target_transform, ignore_metadata=True), \
           TransformDataset(val_dataset, transform=test_transform, target_transform=target_transform, ignore_metadata=True), \
           TransformDataset(test_dataset, transform=test_transform, target_transform=target_transform, ignore_metadata=True), None, None, None, n_class, classnames

def get_wilds_imb_dataset(n_class, data_dir, dataset_name, *args):
    train_dataset, val_dataset, test_dataset = get_wilds_datasets(data_dir, dataset_name)
    return transform_wilds_datasets(n_class, dataset_name, train_dataset, val_dataset, test_dataset)

def get_wilds_shift_dataset(n_class, data_dir, dataset_name, *args):
    train_dataset, val_dataset, test_dataset = get_wilds_datasets(data_dir, dataset_name)
    target_classes = list(range(n_class - 1))
    val_indices_to_keep = torch.where(torch.isin(val_dataset.y_array, torch.tensor(target_classes)))[0]
    val_dataset = Subset(val_dataset, val_indices_to_keep)
    test_indices_to_keep = torch.where(torch.isin(test_dataset.y_array, torch.tensor(target_classes)))[0]
    test_dataset = Subset(test_dataset, test_indices_to_keep)
    return transform_wilds_datasets(n_class, dataset_name, train_dataset, val_dataset, test_dataset)

@register_dataset("iwildcam_imb", LabelType.MULTI_CLASS)
def get_iwildcam_imb_dataset(n_class, data_dir, *args):
    return get_wilds_imb_dataset(n_class, data_dir, "iwildcam", *args)

@register_dataset("fmow_imb", LabelType.MULTI_CLASS)
def get_fmow_imb_dataset(n_class, data_dir, *args):
    return get_wilds_imb_dataset(n_class, data_dir, "fmow", *args)

@register_dataset("iwildcam_shift", LabelType.MULTI_CLASS)
def get_iwildcam_shift_dataset(n_class, data_dir, *args):
    return get_wilds_shift_dataset(n_class, data_dir, "iwildcam", *args)

@register_dataset("fmow_shift", LabelType.MULTI_CLASS)
def get_fmow_shift_dataset(n_class, data_dir, *args):
    return get_wilds_shift_dataset(n_class, data_dir, "fmow", *args)

@register_dataset("iwildcam", LabelType.MULTI_CLASS)
def get_iwildcam_dataset(data_dir, *args):
    full_classnames = get_classnames("iwildcam")
    return get_iwildcam_imb_dataset(len(full_classnames), data_dir, *args)

@register_dataset("fmow", LabelType.MULTI_CLASS)
def get_fmow_dataset(data_dir, *args):
    full_classnames = get_classnames("fmow")
    return get_fmow_imb_dataset(len(full_classnames), data_dir, *args)


if __name__ == "__main__":
    pass
