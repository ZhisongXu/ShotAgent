import os
from PIL import Image
import shutil

def get_image_names(directory):
    image_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_names.append(file)
    return image_names

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

if __name__ == "__main__":
    directory = '/home/data1/NTIRE2025/ShortformUGCSR/submissions/HAT_l_4x_imagenet_psnr/synthetic/HAT_SRx4_ImageNet-pretrain_NTIRE_25_UGC_syn/visualization/test'
    image_names = get_image_names(directory)
    if image_names:
        first_image_path = os.path.join(directory, image_names[0])
        image_size = get_image_size(first_image_path)
        print(f"First image name: {image_names[0]}")
        print(f"First image size: {image_size}")
        destination_directory = '/home/data1/NTIRE2025/ShortformUGCSR/submissions/HAT_l_4x_imagenet_psnr/synthetic'
        os.makedirs(destination_directory, exist_ok=True)

        for image_name in image_names:
            if '_HAT_SRx4_ImageNet-pretrain_NTIRE_25_UGC_syn' in image_name:
                new_image_name = image_name.replace('_HAT_SRx4_ImageNet-pretrain_NTIRE_25_UGC_syn', '')
            source_path = os.path.join(directory, image_name)
            destination_path = os.path.join(destination_directory, new_image_name)
            shutil.copy(source_path, destination_path)
            print(f"Copied {new_image_name} to {destination_directory}")

            # Resize the image (downscale by 2)
            # with Image.open(destination_path) as img:
            #     new_size = (img.width // 2, img.height // 2)
            #     resized_img = img.resize(new_size, Image.LANCZOS)
            #     resized_img.save(destination_path)
            #     print(f"Resized {new_image_name} to {new_size}")