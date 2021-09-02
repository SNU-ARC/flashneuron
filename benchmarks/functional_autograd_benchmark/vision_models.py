import torch
from torch import Tensor
import torchvision_models as models

from utils import extract_weights, load_weights, GetterReturnType

from typing import cast

def get_resnet18(device: torch.device) -> GetterReturnType:
    N = 32
    model = models.resnet18(pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand([N, 3, 224, 224], device=device)
    labels = torch.rand(N, device=device).mul(10).long()

    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    return forward, params

def get_fcn_resnet(device: torch.device) -> GetterReturnType:
    N = 8
    criterion = torch.nn.MSELoss()
    model = models.fcn_resnet50(pretrained=False, pretrained_backbone=False)
    model.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand([N, 3, 480, 480], device=device)
    # Given model has 21 classes
    labels = torch.rand([N, 21, 480, 480], device=device)

    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(inputs)['out']

        loss = criterion(out, labels)
        return loss

    return forward, params

def get_detr(device: torch.device) -> GetterReturnType:
    # All values below are from CLI defaults in https://github.com/facebookresearch/detr
    N = 2
    num_classes = 91
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    model = models.DETR(num_classes=num_classes, hidden_dim=hidden_dim, nheads=nheads,
                        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
    losses = ['labels', 'boxes', 'cardinality']
    eos_coef = 0.1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
    matcher = models.HungarianMatcher(1, 5, 2)
    criterion = models.SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=eos_coef, losses=losses)

    model = model.to(device)
    criterion = criterion.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand(N, 3, 800, 1200, device=device)
    labels = []
    for idx in range(N):
        targets = {}
        n_targets: int = int(torch.randint(5, 10, size=tuple()).item())
        label = torch.randint(5, 10, size=(n_targets,))
        targets["labels"] = label
        boxes = torch.randint(100, 800, size=(n_targets, 4))
        for t in range(n_targets):
            if boxes[t, 0] > boxes[t, 2]:
                boxes[t, 0], boxes[t, 2] = boxes[t, 2], boxes[t, 0]
            if boxes[t, 1] > boxes[t, 3]:
                boxes[t, 1], boxes[t, 3] = boxes[t, 3], boxes[t, 1]
        targets["boxes"] = boxes.float()
        labels.append(targets)

    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        weight_dict = criterion.weight_dict
        final_loss = cast(Tensor, sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict))
        return final_loss

    return forward, params
