
import torch as th
from torch import nn
from torchvision import models, transforms

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        vgg = models.vgg19(pretrained=True).features.eval()
        
        
        for i, layer in enumerate(vgg):
            if isinstance(layer, th.nn.MaxPool2d):
                vgg[i] = th.nn.AvgPool2d(kernel_size=2)

        self.vgg = vgg

    def forward(self, x):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = th.stack([normalize(img) for img in x])  
        features = []
        
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in [1, 3, 13, 22]:  
                features.append(x)

        
        features = [f.flatten(start_dim=1) for f in features]
        return th.cat(features, dim=1)

def compute_vgg_loss(render_image, original_image):
    
    if render_image.shape != original_image.shape:
        raise ValueError("Input images must have the same shape.")
    
    
    vgg_extractor = VGGFeatureExtractor().to(render_image.device)

    
    render_features = vgg_extractor(render_image)
    original_features = vgg_extractor(original_image)

    
    loss_fn = nn.L1Loss()
    vgg_loss = loss_fn(render_features, original_features)

    return vgg_loss
