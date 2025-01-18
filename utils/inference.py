
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from safetensors.torch import load_file

sys.path.append('/home/hpc/vlgm/vlgm103v/genmatpro/idea/')

from models.surfacenet import SurfaceNet  
from my_datasets.surface_dataset import SurfaceDataset  
# Define paths and parameters
metadata_file = "/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/Epoch_2/live_test/input/"
logdir = Path("/home/hpc/vlgm/vlgm103v/genmatpro/idea/exps/default/checkpoints/checkpoint_1/")
checkpoint_path = logdir / "model.safetensors"
output_dir = Path("/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/Epoch_2/live_test/output_1")
sketch_dir=Path("/home/hpc/vlgm/vlgm103v/genmatpro/Evaluation/Epoch_2/live_test/output_idea2/")
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def save_output_maps(output_maps, index):
    output_order = ["normal", "diffuse", "roughness", "specular"]
    output_filenames = [f"{name}_{index}" for name in output_order]
    
    for map_name, file_name in zip(output_order, output_filenames):
        if map_name not in output_maps:
            print(f"Warning: {map_name} map missing for index {index}")
            continue

        output_map = output_maps[map_name]
        output_map_np = output_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_map_np = (output_map_np - output_map_np.min()) / (output_map_np.max() - output_map_np.min())
        output_map_image = (output_map_np * 255).astype("uint8")

        if output_map_image.ndim == 2:
            mode = "L"
        elif output_map_image.shape[2] == 3:
            mode = "RGB"
        elif output_map_image.shape[2] == 1:
            output_map_image = output_map_image.squeeze(-1)
            mode = "L"
        else:
            raise ValueError(f"Unexpected shape for output map: {output_map_image.shape}")

        output_image = Image.fromarray(output_map_image, mode=mode)
        save_path = output_dir / f"{file_name}.png"
        output_image.save(save_path)
        print(f"Saved {map_name} map to {save_path}")


if __name__ == "__main__":
    dataset = SurfaceDataset(dset_dir=metadata_file,sketch_dir=sketch_dir, load_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SurfaceNet().to(device)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        weights = load_file(checkpoint_path)
        model.load_state_dict(weights)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model.eval()

    for i, data in enumerate(dataloader):
        input_tensor = data["render"].to(device)

        sketch_tensor = data["sketch"].to(device)


        image_index = i + 1  

        print(f"Processing Image_{image_index}...")
        with torch.no_grad():
            output_maps = model(input_tensor,sketch_tensor)

        
        if torch.std(output_maps["normal"]) < torch.std(output_maps["diffuse"]):
            print("Swapping diffuse and normal maps based on statistics...")
            output_maps["diffuse"], output_maps["normal"] = output_maps["normal"], output_maps["diffuse"]

        save_output_maps(output_maps, image_index)
