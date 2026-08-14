import cv2
import numpy as np
import os
import argparse

def process_images(original_folder, enhanced_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    for filename in os.listdir(original_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            original_path = os.path.join(original_folder, filename)
            enhanced_path = os.path.join(enhanced_folder, filename)

            # Read the original and enhanced images
            original_image = cv2.imread(original_path)
            enhanced_image = cv2.imread(enhanced_path)

            if original_image is None or enhanced_image is None:
                print(f"Skipping {filename}, unable to read the file.")
                continue

            # Ensure both images have the same dimensions
            if original_image.shape != enhanced_image.shape:
                print(f"Skipping {filename}, size mismatch.")
                continue

            # Compute Laplacian high-frequency details for each channel
            laplacian_b = cv2.Laplacian(original_image[:, :, 0], cv2.CV_64F)
            laplacian_g = cv2.Laplacian(original_image[:, :, 1], cv2.CV_64F)
            laplacian_r = cv2.Laplacian(original_image[:, :, 2], cv2.CV_64F)

            # Convert the result to uint8 format
            high_freq_detail_b = cv2.convertScaleAbs(laplacian_b)
            high_freq_detail_g = cv2.convertScaleAbs(laplacian_g)
            high_freq_detail_r = cv2.convertScaleAbs(laplacian_r)

            # Merge the three channels back
            high_freq_detail = cv2.merge([high_freq_detail_b, high_freq_detail_g, high_freq_detail_r])

            # Blend high-frequency details into the enhanced image
            fused_image = cv2.addWeighted(enhanced_image, 1, high_freq_detail, 0.2, 0)

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, fused_image)
            print(f"Processed: {filename}")

    print("All images have been processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images by fusing high-frequency details from the original image into the enhanced image.")
    parser.add_argument("--original", type=str, required=True, help="Path to the original image folder")
    parser.add_argument("--enhanced", type=str, required=True, help="Path to the enhanced image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to the output image folder")

    args = parser.parse_args()

    process_images(args.original, args.enhanced, args.output)
