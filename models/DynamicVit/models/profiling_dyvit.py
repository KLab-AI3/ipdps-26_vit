import time
import torch
import torch.nn as nn
from dyvit import VisionTransformerDiffPruning

@torch.no_grad()
def measure_block_and_model_latency(image_size=224, batch_size=1, use_fp16=True):
    print(f"\n🧱 Measuring per-block latency for image size {image_size}x{image_size}...")

    device = torch.device("cuda")
    dtype = torch.float16 if use_fp16 else torch.float32

    model = VisionTransformerDiffPruning(
        img_size=image_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        pruning_loc=[3, 6, 9],           # ✅ 用pruning_loc
        token_ratio=[0.7, 0.49, 0.34],   # ✅ 用token_ratio
        num_classes=1000
    ).to(device).eval().requires_grad_(False)

    if use_fp16:
        model = model.half()

    x = torch.randn((batch_size, 3, image_size, image_size), device=device, dtype=dtype)

    # Patch Embedding
    x = model.patch_embed(x)
    if model.cls_token is not None:
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)

    # Measure each block
    for i, blk in enumerate(model.blocks):
        torch.cuda.synchronize()
        start = time.time()

        out = blk(x, policy=None)
        x = out[0] if isinstance(out, (tuple, list)) else out

        torch.cuda.synchronize()
        end = time.time()
        print(f"⏱️ Block {i:02d} latency: {(end - start)*1000:.3f} ms")

    # Measure full model latency
    print(f"\n🚀 Measuring end-to-end latency for DyViT-Tiny...")
    x_full = torch.randn((batch_size, 3, image_size, image_size), device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.time()
    _ = model(x_full)
    torch.cuda.synchronize()
    end = time.time()
    print(f"✅ End-to-end latency: {(end - start)*1000:.3f} ms")



if __name__ == "__main__":
    # 你可以自定义 resolution 和是否使用 fp16
    resolutions = [224, 512, 1024]
    for res in resolutions:
        measure_block_and_model_latency(image_size=res, batch_size=1, use_fp16=True)
