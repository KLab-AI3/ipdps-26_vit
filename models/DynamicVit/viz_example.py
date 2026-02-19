import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.dylvvit import LVViTDiffPruning

# build transforms
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

t_resize_crop = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
])

t_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# build model
BASE_RATE = 0.7
KEEP_RATE = [BASE_RATE, BASE_RATE ** 2, BASE_RATE ** 3]
PRUNING_LOC = [5, 10, 15]
CKPT_PATH = 'pretrained/dynamic-vit_lv-m_r0.7.pth'

model = LVViTDiffPruning(
    patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
    p_emb='4_2',skip_lam=2., return_dense=True, mix_token=True,
    pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, viz_mode=True,
)

checkpoint = torch.load(CKPT_PATH, map_location='cpu')['model']
model.load_state_dict(checkpoint)

def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices
def gen_masked_tokens(tokens, indices, alpha=0.2):
    indices = [i for i in range(196) if i not in indices]
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens

def recover_image(tokens):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(14, 14, 16, 16, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

def gen_visualization(image, decisions):
    keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    image_tokens = image.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, keep_indices[i]))
        for i in range(3)
    ]
    viz = np.concatenate([image] + stages, axis=1)
    return viz

# visualization
# use a random image from IMAGE_ROOT

IMAGE_ROOT = 'imgs'
image_paths = glob.glob(f'imgs/*.jpg')
idx = np.random.randint(len(image_paths))
image_path = image_paths[idx]
image = Image.open(image_path)

image = t_resize_crop(image)
im_tensor = t_to_tensor(image).unsqueeze(0)

device = 'cpu'
model.to(device)
model.eval()
im_tensor = im_tensor.to(device)
with torch.cuda.amp.autocast():
    output, decisions = model(im_tensor)
decisions = [decisions[i][0][0].cpu().numpy() for i in range(3)]
viz = gen_visualization(image, decisions)

plt.figure(figsize=(20, 5))
plt.imshow(viz)
plt.axis('off')