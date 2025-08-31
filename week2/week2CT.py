import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import sys
from pathlib import Path


def download_image(url):
    """Download image from URL and convert to OpenCV format"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # Decode image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
            
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def display_images(images, titles, figsize=(15, 10)):
    """Display multiple images in a grid"""
    n_images = len(images)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        
        # Convert BGR to RGB for matplotlib display if image has 3 channels
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            plt.imshow(image, cmap='gray')
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    import sys
    from pathlib import Path

    # Local image path (your hardcoded one from development machine)
    image_path = "/home/jason/projects/openvc/week2/kitten_photo.jpg"
    print(f"Loading image from: {image_path}")

    # Try the hardcoded path first
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # If that fails, try to use the bundled file from PyInstaller build
    if image is None:
        try:
            base = Path(getattr(sys, "_MEIPASS", Path.cwd()))
            bundled = base / "kitten_photo.jpg"   # we'll add this via --add-data
            print(f"Hardcoded path failed. Trying bundled image: {bundled}")
            image = cv2.imread(str(bundled), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Bundled image load error: {e}")
            image = None

    if image is None:
        print(f"Error: Could not load image from {image_path} or from bundled copy.")
        print("Please check that the file exists or that the binary was built with --add-data.")
        return

    print(f"Successfully loaded image!")
    print(f"Original image shape: {image.shape}")
    print(f"Image dimensions: {image.shape[0]} x {image.shape[1]} x {image.shape[2]}")

    # Step 1: Extract each color channel separately
    print("\nStep 1: Extracting color channels...")
    blue_channel = image[:, :, 0]   # Extract blue channel
    green_channel = image[:, :, 1]  # Extract green channel
    red_channel = image[:, :, 2]    # Extract red channel
    print(f"Blue channel shape: {blue_channel.shape}")
    print(f"Green channel shape: {green_channel.shape}")
    print(f"Red channel shape: {red_channel.shape}")

    # Display original image and extracted channels
    images_step1 = [image, blue_channel, green_channel, red_channel]
    titles_step1 = ['Original Image', 'Blue Channel', 'Green Channel', 'Red Channel']
    display_images(images_step1, titles_step1)

    # Step 2: Merge channels back into colored 3D image
    print("\nStep 2: Merging channels back into colored image...")
    merged_image_cv2 = cv2.merge([blue_channel, green_channel, red_channel])
    merged_image_np = np.stack([blue_channel, green_channel, red_channel], axis=2)
    merged_image_manual = np.zeros_like(image)
    merged_image_manual[:, :, 0] = blue_channel
    merged_image_manual[:, :, 1] = green_channel
    merged_image_manual[:, :, 2] = red_channel
    print(f"cv2.merge and original identical: {np.array_equal(merged_image_cv2, image)}")
    print(f"numpy stack and original identical: {np.array_equal(merged_image_np, image)}")
    print(f"manual assignment and original identical: {np.array_equal(merged_image_manual, image)}")
    images_step2 = [image, merged_image_cv2]
    titles_step2 = ['Original Image', 'Merged Back (Reconstructed)']
    display_images(images_step2, titles_step2, figsize=(12, 6))
    images_methods = [merged_image_cv2, merged_image_np, merged_image_manual]
    titles_methods = ['Merged (cv2.merge)', 'Merged (np.stack)', 'Merged (manual)']
    display_images(images_methods, titles_methods)

    # Step 3: Create GRB image (swapping red and green channels)
    print("\nStep 3: Creating GRB image (swapping red and green channels)...")
    grb_image = cv2.merge([blue_channel, red_channel, green_channel])
    grb_image_np = np.zeros_like(image)
    grb_image_np[:, :, 0] = blue_channel
    grb_image_np[:, :, 1] = red_channel
    grb_image_np[:, :, 2] = green_channel
    print(f"GRB images identical: {np.array_equal(grb_image, grb_image_np)}")
    images_step3 = [image, grb_image]
    titles_step3 = ['Original (BGR)', 'Modified (GRB - Red↔Green swapped)']
    display_images(images_step3, titles_step3, figsize=(12, 6))

    # Additional analysis
    print("\nAdditional Analysis:")
    print("Effect of swapping red and green channels:")
    print("- Areas that were red will appear green")
    print("- Areas that were green will appear red")
    print("- Blue areas remain unchanged")
    print("- This creates a red-green color swap effect")

    # Channel visualizations
    print("\nCreating individual channel visualizations...")
    blue_only = np.zeros_like(image)
    blue_only[:, :, 0] = blue_channel
    green_only = np.zeros_like(image)
    green_only[:, :, 1] = green_channel
    red_only = np.zeros_like(image)
    red_only[:, :, 2] = red_channel
    images_channels = [blue_only, green_only, red_only]
    titles_channels = ['Blue Channel Only', 'Green Channel Only', 'Red Channel Only']
    display_images(images_channels, titles_channels)

    # Save outputs
    print("\nSaving processed images...")
    cv2.imwrite('original_image.png', image)
    cv2.imwrite('merged_image.png', merged_image_cv2)
    cv2.imwrite('grb_swapped_image.png', grb_image)
    cv2.imwrite('blue_channel.png', blue_channel)
    cv2.imwrite('green_channel.png', green_channel)
    cv2.imwrite('red_channel.png', red_channel)
    print("Processing complete! All images saved.")
    print("Saved files:")
    print("- original_image.png (original kitten image)")
    print("- merged_image.png (Step 2: channels merged back)")
    print("- grb_swapped_image.png (Step 3: red-green swapped)")
    print("- blue_channel.png (individual blue channel)")
    print("- green_channel.png (individual green channel)")
    print("- red_channel.png (individual red channel)")


if __name__ == "__main__":
    main()