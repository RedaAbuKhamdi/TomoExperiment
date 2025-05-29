import os
import sys
import numpy as np
from PIL import Image
import trimesh

def stl_to_png_slices(stl_path, out_folder, resolution=128):
    # Load the STL mesh
    mesh = trimesh.load(stl_path)

    # Normalize the mesh: move to origin and scale to target resolution
    mesh.apply_translation(-mesh.bounds[0])
    scale = resolution / max(mesh.extents)
    mesh.apply_scale(scale)

    # Create voxel grid with a pitch of 1.0 (you can reduce for higher resolution)
    voxelized = mesh.voxelized(pitch=1.0)

    # Fill the interior to make a solid volume
    filled = voxelized.fill()
    matrix = filled.matrix  # 3D numpy array of booleans (True = filled voxel)

    # Ensure output folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Slice along Z-axis and save each slice as a grayscale PNG
    for z in range(matrix.shape[2]):
        slice_img = (matrix[:, :, z] * 255).astype(np.uint8)
        img = Image.fromarray(slice_img, mode='L')
        img.save(os.path.join(out_folder, f"slice_{z:03d}.png"))

    print(f"Saved {matrix.shape[2]} slices with shape {matrix.shape}")

def iterate_images_directory(directory : str):
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.png'):
            continue
        path = os.path.join(directory, fname)
        
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.uint8)
        
        yield arr, path

def invert_binary_pngs(dir_path):
    for image, path in iterate_images_directory(dir_path):
        inv = 255 - image
        Image.fromarray(inv).save(path)

def mash_pngs(dir_path, other_path):
    original_images_iterator = iterate_images_directory(dir_path)
    other_images_iterator = iterate_images_directory(other_path)

    for image, path in original_images_iterator:
        other_image, _ = next(other_images_iterator)
        mask = (other_image > 0)
        image[mask == False] = 0
        Image.fromarray(image).save(path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <operation> <args>")
        sys.exit(1)
    if sys.argv[1] == "invert":
        invert_binary_pngs(sys.argv[2])
    elif sys.argv[1] == "mash":
        mash_pngs(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "stl2png":
        stl_to_png_slices(sys.argv[2], sys.argv[3])
    else:
        print("Unknown operation - " + sys.argv[1])