import torch
import os
from PIL import Image
from torchvision import transforms
from model import AetherLiftNet  # adjust if your model file name differs

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
# Load checkpoint first to extract config (matches training architecture)
ckpt = torch.load("models/best.pth", map_location=device)
cfg = ckpt.get("cfg", {})

model = AetherLiftNet(
    base_ch=cfg.get("base_ch", 48),
    num_res_blocks=cfg.get("num_res_blocks", 4)
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print("Loaded model with config:", cfg)

# Paths
input_dir = "data/val/hazy"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# Inference
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    out_img = out.squeeze(0).cpu().clamp(0, 1)
    out_img = to_pil(out_img)

    out_img.save(os.path.join(output_dir, img_name))

print("Done. Check results/")
