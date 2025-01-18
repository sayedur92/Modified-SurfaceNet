from render1.util import *
import torch as th

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

import numpy as np
from PIL import Image

device = th.device("cuda" if th.cuda.is_available() else "cpu")
eps = 1e-6

def tex2map(tex):
    albedo = ((tex[:, 0:3, :, :].clamp(-1, 1) + 1) / 2) ** 2.2

    normal_x = tex[:, 3, :, :].clamp(-1, 1)
    normal_y = tex[:, 4, :, :].clamp(-1, 1)
    normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1 - eps)
    normal_z = (1 - normal_xy).sqrt()
    normal = th.stack((normal_x, normal_y, normal_z), 1)
    normal = normal.div(normal.norm(2.0, 1, keepdim=True))

    rough = ((tex[:, 5, :, :].clamp(-0.3, 1) + 1) / 2) ** 2.2
    rough = rough.clamp(min=eps).unsqueeze(1).expand(-1, 3, -1, -1)

    if tex.size(1) == 9:
        specular = ((tex[:, 6:9, :, :].clamp(-1, 1) + 1) / 2) ** 2.2
        return albedo, normal, rough, specular

    return albedo, normal, rough

class Microfacet:
    def __init__(self, res, size, f0=0.04):
        self.res = res
        self.size = size
        self.f0 = f0
        self.eps = eps
        self.initGeometry()

    def initGeometry(self):
        tmp = th.linspace(-0.5 * self.size, 0.5 * self.size, self.res, device=device)
        y, x = th.meshgrid(tmp, tmp, indexing="ij")
        self.pos = th.stack((x, -y, th.zeros_like(x)), 2)
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def normalize(self, vec):
        return vec / (vec.norm(2.0, 1, keepdim=True) + self.eps)

    def getDir(self, pos):
        if not isinstance(pos, th.Tensor):
            pos = th.tensor(pos, dtype=th.float32).to(device)
        vec = (pos - self.pos).permute(2, 0, 1).unsqueeze(0).expand(self.N, -1, -1, -1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1, 3, -1, -1)

    def eval(self, textures, lightPos, cameraPos, light):
        self.N = textures.size(0)
        isSpecular = textures.size(1) == 9
        if isSpecular:
            albedo, normal, rough, specular = tex2map(textures)
        else:
            albedo, normal, rough = tex2map(textures)
        light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)

        v, _ = self.getDir(cameraPos)
        l, dist_l_sq = self.getDir(lightPos)
        h = self.normalize(l + v)

        n_dot_v = th.sum(normal * v, dim=1, keepdim=True).clamp(min=0)
        n_dot_l = th.sum(normal * l, dim=1, keepdim=True).clamp(min=0)
        n_dot_h = th.sum(normal * h, dim=1, keepdim=True).clamp(min=0)
        v_dot_h = th.sum(v * h, dim=1, keepdim=True).clamp(min=0)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough**2)
        F = self.Fresnel_S(v_dot_h, specular) if isSpecular else self.Fresnel(v_dot_h, self.f0)
        G = self.Smith(n_dot_v, n_dot_l, rough**2)

        f1 = albedo / np.pi
        if isSpecular:
            f1 *= (1 - specular)
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        f = f1 + f2
        img = f * geom * light

        return img.clamp(0, 1)

    def GGX(self, cos_h, alpha):
        a2 = alpha**2
        return a2 / (np.pi * ((cos_h**2) * (a2 - 1) + 1)**2 + self.eps)

    def Fresnel(self, cos, f0):
        return f0 + (1 - f0) * ((1 - cos)**5)

    def Fresnel_S(self, cos, specular):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        k = alpha * 0.5 + self.eps
        return (n_dot_v / (n_dot_v * (1 - k) + k)) * (n_dot_l / (n_dot_l * (1 - k) + k))

def renderTexFromTensor(output_tensor, 
                        res=256, 
                        size=1.0, 
                        lp=[1.0, 1.0, 1.0], 
                        cp=[0.0, 0.0, 3.0], 
                        L=[10.0, 10.0, 10.0],
                        device="cuda" if th.cuda.is_available() else "cpu"):
    # Ensure lp and cp are tensors
    lp = th.tensor(lp, dtype=th.float32).to(device) if not isinstance(lp, th.Tensor) else lp
    cp = th.tensor(cp, dtype=th.float32).to(device) if not isinstance(cp, th.Tensor) else cp
    L = th.tensor(L, dtype=th.float32).to(device) if not isinstance(L, th.Tensor) else L

    if isinstance(output_tensor, dict):
        if all(key in output_tensor for key in ["diffuse", "normal", "roughness", "specular"]):
            diffuse = output_tensor["diffuse"]
            normal = output_tensor["normal"]
            roughness = output_tensor["roughness"]
            specular = output_tensor["specular"]
            textures = th.cat([diffuse, normal, roughness, specular], dim=1).to(device)
        else:
            raise KeyError("The provided dictionary must contain 'diffuse', 'normal', 'roughness', and 'specular' keys.")
    elif isinstance(output_tensor, th.Tensor):
        textures = output_tensor.to(device)
    else:
        raise TypeError("output_tensor must be a dictionary containing texture maps or a torch.Tensor.")

    try:
        renderObj = Microfacet(res=res, size=size)
        rendered_images = []
        
        # Process batch
        for i in range(textures.size(0)):
            rendered_image = renderObj.eval(
                textures[i].unsqueeze(0), lp, cp, L
            )
            rendered_image = gyApplyGamma(
                gyTensor2Array(rendered_image[0, :].permute(1, 2, 0).detach().cpu()), 1 / 2.2
            )
            rendered_images.append(gyArray2PIL(rendered_image))
        
        # Return single image or batch
        return rendered_images if len(rendered_images) > 1 else rendered_images[0]
    except Exception as e:
        raise RuntimeError(f"Rendering failed: {e}")
