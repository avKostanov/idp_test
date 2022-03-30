import torch
from efficientnet_pytorch import EfficientNet

def getModel(weights='weights.pth'):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    for param in model.parameters():
        param.requires_grad = False
    in_features = model._fc.in_features
    model._fc = torch.nn.Linear(in_features, 10)
    model.load_state_dict(torch.load(
        weights, map_location=torch.device('cpu')))
    return model