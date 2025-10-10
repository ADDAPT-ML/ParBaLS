# Code related to zeroshot classifier is mainly ported from
# https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py
import torch
import torch.nn as nn
from torchvision import transforms

from LabelBench.skeleton.model_skeleton import register_model


class DINO(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain, model_name):

        super(DINO, self).__init__()
        assert pretrain, "DINO only support pretrain model"

        model = torch.hub.load('facebookresearch/dinov2', model_name)
        # https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L65-L70
        dino_mean = (0.485, 0.456, 0.406)
        dino_std = (0.229, 0.224, 0.225)
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) != 3 else x),
            transforms.Normalize(dino_mean, dino_std),
        ])
        self.image_encoder_model = model.float()  # Convert to float to avoid NAN loss when using AdamW.
        self.embed_dim = model.state_dict()["norm.weight"].shape[0]
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output)
        else:
            self.classifier = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.image_encoder_model(imgs)
        else:
            features = self.image_encoder_model(imgs)

        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self, split):
        return self.preprocess_transform


@register_model("dinov2_vits14")
def init_dinov2_vits14(model_config):
    return DINO(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="dinov2_vits14")


@register_model("dinov2_vitb14")
def init_dinov2_vitb14(model_config):
    return DINO(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="dinov2_vitb14")
