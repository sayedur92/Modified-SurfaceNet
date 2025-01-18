import os
import numpy as np
from PIL import Image
import torch
import lpips
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pytorch_msssim import ms_ssim


def normalize_image(img):

    return img / 255.0


def l1_loss(img1, img2):
    return np.mean(np.abs(img1 - img2))


def compute_ssim(img1, img2):
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
    return ssim(img1, img2, data_range=1.0, channel_axis=-1, win_size=win_size)


def compute_msssim(img1, img2):
    img1_t = torch.tensor(img1.transpose(2, 0, 1)).unsqueeze(0)
    img2_t = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0)
    return ms_ssim(img1_t, img2_t, data_range=1.0).item()


def compute_aggregated_metrics(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    prefixes = ["normal", "diffuse", "roughness", "specular"]
    aggregated_results = {}

    transform = transforms.Compose([transforms.ToTensor()])

    for prefix in prefixes:
        input_files = sorted([f for f in os.listdir(input_folder) if f.startswith(prefix)])
        output_files = sorted([f for f in os.listdir(output_folder) if f.startswith(prefix)])

        if len(input_files) != len(output_files):
            print(f"Warning: Mismatch in file count for prefix '{prefix}'.")
            continue

        psnr_values, l1_values, ssim_values, msssim_values, lpips_values = [], [], [], [], []

        for input_file, output_file in zip(input_files, output_files):
            input_path = os.path.join(input_folder, input_file)
            output_path = os.path.join(output_folder, output_file)

            if os.path.isfile(input_path) and os.path.isfile(output_path):
                img1 = np.array(Image.open(input_path).convert("RGB"))
                img2 = np.array(Image.open(output_path).convert("RGB"))

                # Normalize images
                img1 = normalize_image(img1)
                img2 = normalize_image(img2)

                psnr_values.append(psnr(img1, img2, data_range=1.0))
                l1_values.append(l1_loss(img1, img2))
                ssim_values.append(compute_ssim(img1, img2))
                msssim_values.append(compute_msssim(img1, img2))

                img1_lpips = transform(Image.fromarray((img1 * 255).astype("uint8"))).unsqueeze(0).to(device)
                img2_lpips = transform(Image.fromarray((img2 * 255).astype("uint8"))).unsqueeze(0).to(device)
                lpips_values.append(lpips_model(img1_lpips, img2_lpips).item())

        # Aggregate metrics
        aggregated_results[prefix] = {
            "PSNR": np.mean(psnr_values),
            "L1 Loss": np.mean(l1_values),
            "SSIM": np.mean(ssim_values),
            "MS-SSIM": np.mean(msssim_values),
            "LPIPS": np.mean(lpips_values),
        }

    return aggregated_results


if __name__ == "__main__":
    input_folder = "/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/Epoch_2/input_test_gt/"
    output_folder = "/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/Epoch_2/output_test_idea1/"
    aggregated_metrics = compute_aggregated_metrics(input_folder, output_folder)

    for prefix, metrics in aggregated_metrics.items():
        print(f"\nAggregated Metrics for {prefix}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

'''scp -r /c/Users/REZVI/live vlgm103v@tinyx.nhr.fau.de:/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/live_test/input
'''