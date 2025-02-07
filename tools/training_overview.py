import os
from PIL import Image
import re

def create_training_overview(image_folder, output_gif_path='../05_Results/training_gifs', framerate=2):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    #image_files.sort()  # Sort files if needed

    # Function to extract the numeric part from the filename after the last underscore
    def extract_number(filename):
        match = re.search(r'_(\d+)\.', filename)
        return int(match.group(1)) if match else float('inf')  # Return inf if no number found

    # Sort files numerically based on the extracted numbers
    image_files.sort(key=extract_number)

    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Convert the framerate (frames per second) to a duration per frame
    duration_per_frame = 1000 / framerate  # duration in milliseconds

    # Save images as GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration_per_frame, loop=0)
    print(f"GIF saved as {output_gif_path}")

