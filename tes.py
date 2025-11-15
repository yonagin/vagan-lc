# Cell 1: 导入包（最小化）
%cd vq
!git pull
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.models_vq import VQModel
from PIL import Image
import albumentations
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cell 2: 加载配置
def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_container(config, resolve=True))
    return config

# Cell 3: 测试数据集
class TestDataset(Dataset):
    def __init__(self, path, size=64, max_size=-1):
        self.size = size
        self.images = [os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # 限制数据集大小
        if max_size > 0 and max_size < len(self.images):
            self.images = self.images[:max_size]
            print(f"限制数据集大小为: {max_size} 张图片")
        
        self._length = len(self.images)
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return torch.from_numpy(example), self.images[i]

# Cell 4: 参数类
class Args:
    def __init__(self):
        self.batch_size = 8
        self.output_dir = './output_test_vq'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.seed = 0
        self.num_workers = 0  
        self.pin_mem = False
        self.dataset_path = './img'
        self.vq_config_path = './vqgan_configs/vq-f16.yaml'
        self.model_path = './mbin/vqgan-lc-100K-f16-dim8.pth'
        self.embed_dim = 8
        self.n_vision_words = 100000
        self.quantizer_type = 'norm_vq'
        self.tuning_codebook = 0
        self.stage = 1
        self.use_cblinear = 1
        self.rate_q = 1.0
        self.rate_p = 1.0
        self.rate_d = 1.0
        self.disc_start = 10000
        self.image_size = 256
        self.local_embedding_path = './mbin/codebook-100K.pth'
        self.dataset = 'custom'
        self.test_dataset_size = -1  # -1表示使用全部数据，正整数表示限制数据集大小
        self.candidate_ratio = 0.1

args = Args()

# Cell 5: 主逻辑
print("开始测试...")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset_test = TestDataset(args.dataset_path, size=args.image_size, max_size=args.test_dataset_size)
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=False)

config = load_config(args.vq_config_path, display=True)
model = VQModel(args=args, **config.model.params)
sd = torch.load(args.model_path, map_location="cpu")
if 'state_dict' in sd:
    sd = sd['state_dict']
elif 'model' in sd:
    sd = sd['model']
missing, unexpected = model.load_state_dict(sd, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.to(DEVICE)
model.eval()

os.makedirs(args.output_dir, exist_ok=True)
recons_save_dir = os.path.join(args.output_dir, "recons")
os.makedirs(recons_save_dir, exist_ok=True)

mse_total = 0.0
psnr_total = 0.0
num_images = 0
token_freq = torch.zeros(args.n_vision_words).to(DEVICE)
count = 0
start_time = time.time()

def compute_psnr(mse):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

for data_iter_step, (images, paths) in enumerate(data_loader):
    images = images.to(DEVICE)
    b = images.shape[0]

    with torch.no_grad():
        xrec = model(images, None, data_iter_step, step=0, is_val=True)
        # 从encode获取tk_labels
        _, _, [_, _, tk_labels] = model.encode(images)

    # 计算MSE
    mse = F.mse_loss(images, xrec, reduction='none').mean(dim=[1,2,3])
    mse_total += torch.sum(mse).item()
    num_images += b

    # PSNR (简单版，假设data_range=2 因为[-1,1])
    psnr = 10 * torch.log10(4 / (mse + 1e-8))  # (max_val=2)^2 / mse
    psnr_total += torch.sum(psnr).item()

    # Token freq
    tk_index_one_hot = F.one_hot(tk_labels.view(-1), num_classes=args.n_vision_words)
    tk_index_num = torch.sum(tk_index_one_hot, dim=0)
    token_freq += tk_index_num

    # 保存重建
    xrec = torch.clamp(xrec, -1, 1)
    save_xrec = (xrec + 1) / 2.0  # [0,1]
    for i in range(b):
        orig_name = os.path.basename(paths[i])
        recon_path = os.path.join(recons_save_dir, f"recon_{orig_name}")
        recon_img = np.uint8(save_xrec[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)
        plt.imsave(recon_path, recon_img)
        count += 1

# 汇总
avg_mse = mse_total / num_images
avg_psnr = psnr_total / num_images
efficient_token = (token_freq > 0).sum().item()

# 获取峰值显存使用情况
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
else:
    peak_memory = 0

print(f"平均 MSE: {avg_mse:.4f} | 平均 PSNR: {avg_psnr:.4f} | 有效 Token: {efficient_token}")
print(f"峰值显存: {peak_memory:.2f} MB" if peak_memory > 0 else "峰值显存: N/A (CPU模式)")

total_time = time.time() - start_time
print(f'测试时间: {total_time:.2f} 秒')

# 保存结果
with open(os.path.join(args.output_dir, "test_results.txt"), 'w') as f:
    f.write(f"平均 MSE: {avg_mse:.4f}\n")
    f.write(f"平均 PSNR: {avg_psnr:.4f}\n")
    f.write(f"有效 Token: {efficient_token}\n")
    if peak_memory > 0:
        f.write(f"峰值显存: {peak_memory:.2f} MB\n")
    else:
        f.write("峰值显存: N/A (CPU模式)\n")