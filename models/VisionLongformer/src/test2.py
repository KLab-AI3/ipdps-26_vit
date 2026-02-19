import torch
import time
import csv
from models.msvit import MsViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分辨率和 patch 大小 sweep
resolutions = [224, 512, 1024.2048, 4096, 8192]
patch_sizes = [1, 2, 4, 8, 16, 32]

# 构造 arch 字符串（动态 patch）
def generate_arch(patch):
    return f"l1,h1,d48,n1,s1,g1,p{patch},f7_" + \
           f"l2,h3,d96,n1,s1,g1,p{patch},f7_" + \
           f"l3,h3,d192,n9,s0,g1,p{patch},f7_" + \
           f"l4,h6,d384,n1,s0,g0,p{patch},f7"

@torch.no_grad()
def measure_latency(model, input_tensor, warmup=10, iters=30):
    model.eval()
    times = []

    for _ in range(warmup):
        _ = model(input_tensor)

    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    return sum(times) / len(times)

# 写入结果 CSV
csv_file = "res_patch_latency.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Resolution", "Patch Size", "Num Tokens", "Latency (ms)"])

    for res in resolutions:
        for patch in patch_sizes:
            tokens = (res // patch) ** 2
            print(f"\n🧱 Measuring latency for resolution {res}x{res}, patch {patch}...")
            print(f"🔢 Number of tokens: {tokens}")

            try:
                arch = generate_arch(patch)
                model = MsViT(
                    arch=arch,
                    img_size=res,
                    in_chans=3,
                    num_classes=1000,
                    attn_type='longformerhand',
                    avg_pool=False,
                    sw_exact=1,
                    mode=0
                ).to(device).half()

                dummy_input = torch.randn(1, 3, res, res).to(device).half()
                latency = measure_latency(model, dummy_input)
                print(f"⏱️  End-to-end latency: {latency:.3f} ms")
                writer.writerow([res, patch, tokens, f"{latency:.3f}"])

            except Exception as e:
                print(f"❌ Failed at resolution {res}, patch {patch}: {e}")
                writer.writerow([res, patch, tokens, "FAIL"])
