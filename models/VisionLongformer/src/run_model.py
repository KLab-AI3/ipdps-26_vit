# src/run_model.py

from models.msvit import MsViT
import torch

arch = "l1,h1,d48,n1,s1,g1,p4,f7_l2,h3,d96,n1,s1,g1,p2,f7_l3,h3,d192,n9,s0,g1,p2,f7_l4,h6,d384,n1,s0,g0,p2,f7"


model = MsViT(arch=arch, img_size=224, num_classes=1000)

x = torch.randn(1, 3, 224, 224)
out = model(x)
print("Output shape:", out.shape)
