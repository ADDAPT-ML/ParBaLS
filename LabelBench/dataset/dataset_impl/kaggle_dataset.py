import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from kaggle.api.kaggle_api_extended import KaggleApi
from LabelBench.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType, TransformDataset


@register_dataset("kaggle_tabular", LabelType.MULTI_CLASS)
def get_kaggle_tabular_dataset(dataset_link, data_dir, target_column, *args):
    dataset_name = dataset_link.split('/')[-1]
    dataset_path = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset '{dataset_link}' to '{dataset_path}'...")
        os.makedirs(dataset_path, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_link, path=dataset_path, unzip=True)
        print("Download complete.")
    else:
        print(f"Dataset already exists at '{dataset_path}'.")

    train_path = os.path.join(dataset_path, 'train.csv')
    test_path = os.path.join(dataset_path, 'test.csv')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Reading pre=split files for dataset: '{dataset_link}'")
        train_df = pd.read_csv(train_path)
        test_val_df = pd.read_csv(test_path)
        val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    # TODO: add more datasets here
    else:
        print(f"Applying general splitting logic for dataset: '{dataset_link}'")
        csv_file = next(f for f in os.listdir(dataset_path) if f.endswith('.csv'))
        df = pd.read_csv(os.path.join(dataset_path, csv_file))
        train_df, rem_df = train_test_split(df, train_size=0.6, random_state=42)
        val_df, test_df = train_test_split(rem_df, test_size=0.5, random_state=42)

    id_cols_to_drop = [
        col for col in train_df.columns 
        if (col.lower() == 'id' or col.lower().endswith('_id') or 'unnamed' in col.lower())
        and col != target_column
    ]
    
    if id_cols_to_drop:
        print(f"Automatically detected and removed ID columns: {id_cols_to_drop}")
        train_df = train_df.drop(columns=id_cols_to_drop)
        val_df = val_df.drop(columns=id_cols_to_drop, errors='ignore')
        test_df = test_df.drop(columns=id_cols_to_drop, errors='ignore')

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    classnames = sorted(y_train.dropna().unique())
    n_class = len(classnames)
    target_map = {name: i for i, name in enumerate(classnames)}
    
    y_train_tensor = F.one_hot(torch.tensor([target_map.get(v, -1) for v in y_train]), num_classes=n_class).float()
    y_val_tensor = F.one_hot(torch.tensor([target_map.get(v, -1) for v in y_val]), num_classes=n_class).float()
    y_test_tensor = F.one_hot(torch.tensor([target_map.get(v, -1) for v in y_test]), num_classes=n_class).float()
    
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numerical_strategy = "quantile"
    if numerical_features:
        if numerical_strategy == "quantile":
            print(f"Applying quantile binning to numerical features with 10 bins.")
            preprocessor = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile', subsample=None)

            X_train_num = X_train[numerical_features].fillna(X_train[numerical_features].mean())
            X_val_num = X_val[numerical_features].fillna(X_train[numerical_features].mean())
            X_test_num = X_test[numerical_features].fillna(X_train[numerical_features].mean())

            X_train_num_processed = preprocessor.fit_transform(X_train_num)
            X_val_num_processed = preprocessor.transform(X_val_num)
            X_test_num_processed = preprocessor.transform(X_test_num)

        elif numerical_strategy == 'standard':
            print("Applying standard scaling to numerical features.")
            preprocessor = StandardScaler()

            X_train_num = X_train[numerical_features].fillna(X_train[numerical_features].mean())
            X_val_num = X_val[numerical_features].fillna(X_train[numerical_features].mean())
            X_test_num = X_test[numerical_features].fillna(X_train[numerical_features].mean())

            X_train_num_processed = preprocessor.fit_transform(X_train_num)
            X_val_num_processed = preprocessor.transform(X_val_num)
            X_test_num_processed = preprocessor.transform(X_test_num)
        else:
            raise ValueError("numerical_strategy must be 'standard' or 'quantile'")

    processed_features = []
    for split_idx, X_split in enumerate([X_train, X_val, X_test]):
        if numerical_features:
            num_tensor = torch.tensor([X_train_num_processed, X_val_num_processed, X_test_num_processed][split_idx], dtype=torch.float32)
        else:
            num_tensor = torch.empty((len(X_split), 0), dtype=torch.float32)

        if categorical_features:
            cat_tensors = []
            for col in categorical_features:
                vocab = sorted(X_train[col].dropna().unique())
                cat_map = {cat: vocab_idx for vocab_idx, cat in enumerate(vocab)}
                cat_data = X_split[col].fillna('NA')
                cat_indices = torch.tensor([cat_map.get(val, len(vocab)) for val in cat_data], dtype=torch.long)
                cat_tensor = F.one_hot(cat_indices, num_classes=len(vocab) + 1).float()
                cat_tensors.append(cat_tensor)
            cat_tensor_combined = torch.cat(cat_tensors, dim=1)
        else:
            cat_tensor_combined = torch.empty((len(X_split), 0), dtype=torch.float32)
        
        processed_features.append(torch.cat((num_tensor, cat_tensor_combined), dim=1))

    X_train_tensor, X_val_tensor, X_test_tensor = processed_features
    
    base_train_dataset = DatasetOnMemory(X_train_tensor, y_train_tensor, n_class)
    base_val_dataset = DatasetOnMemory(X_val_tensor, y_val_tensor, n_class)
    base_test_dataset = DatasetOnMemory(X_test_tensor, y_test_tensor, n_class)

    train_dataset = TransformDataset(base_train_dataset)
    val_dataset = TransformDataset(base_val_dataset)
    test_dataset = TransformDataset(base_test_dataset)

    return train_dataset, val_dataset, test_dataset, None, None, None, n_class, list(classnames)

if __name__ == "__main__":
    train_ds, val_ds, test_ds, _, _, _, n_class, classnames = get_kaggle_tabular_dataset(
        dataset_link='teejmahal20/airline-passenger-satisfaction',
        data_dir='./data',
        target_column='satisfaction'
    )
    
    print(f"Number of classes: {n_class}, Class names: {classnames}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    print(f"Returned dataset type: {type(train_ds)}")

    loader = DataLoader(train_ds, batch_size=4)
    x_batch, y_batch = next(iter(loader))
    
    print(f"Feature batch shape: {x_batch.size()}")
    print(f"Label batch shape: {y_batch.size()}")
    print("Sample feature tensor:", x_batch[0])
    print("Sample label tensor:", y_batch[0])
