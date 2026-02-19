import torch
import time
from models.msvit import MsViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用你指定的 arch
arch = "l1,h1,d48,n1,s1,g1,p4,f7_l2,h3,d96,n1,s1,g1,p2,f7_l3,h3,d192,n9,s0,g1,p2,f7_l4,h6,d384,n1,s0,g0,p2,f7"

# 测试分辨率
test_resolutions = [224, 512, 1024, 2048, 4096, 8192]

@torch.no_grad()
def measure_latency(model, input_tensor, warmup=10, iters=50):
    model.eval()
    times = []

    # warmup
    for _ in range(warmup):
        _ = model(input_tensor)

    # measure
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)

if __name__ == "__main__":
    for res in test_resolutions:
        print(f"\n🧱 Measuring latency for resolution {res}x{res} using custom arch...")
        try:
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
            print(f"✅ Avg end-to-end latency: {latency:.3f} ms")

        except AssertionError as e:
            print(f"❌ Failed: {e}")
