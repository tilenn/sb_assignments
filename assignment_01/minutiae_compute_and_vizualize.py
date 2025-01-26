import cv2
import subprocess
import numpy as np
import sys
import os

# IMPORTANT:
# Mindtct most likely won't work on TIFF images and will silently fail.
#   In cases like that transform images to PNG first.
#   One way of solving this is using magick: "magick some_img.tif some_img.png", before running this script.

BINARY_PATH = "mindtct"     # If your PATH is set correctly, you can leave it as is, regardless of the operating system.
CONTRAST_ENHANCEMENT = True # Set flag for optional contrast enhancement of the image
DEFAULT_OUT_DIR = "out"

# Update to handle positional arguments
default_image_path = 'example.png'
if len(sys.argv) > 1:
    image_path = sys.argv[1] 
else:
    print("No file was given, defaulting to:", default_image_path)
    image_path = default_image_path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Fatal. No file was given or the file '{image_path}' does not exist.")

base_path = os.path.splitext(image_path)[0]
base_name = os.path.basename(base_path)

# Update results_location based on the second positional argument
out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
os.makedirs(out_dir, exist_ok=True) if out_dir else None


def read_minutiae(result_location):
    # Function to parse ".min" file
    result_location = result_location + '.min'

    if not os.path.isfile(result_location):
        raise FileNotFoundError(f"Fatal. File '{result_location}' does not exist, mindtct probably failed to compute minutiae.")

    with open(result_location, "r") as handle:
        content = handle.readlines()

    minutiae_list = []
    for i, line in enumerate(content):
       
        # Skip header
        if i < 4:
            continue

        # Split by ":"
        results = [x.strip() for x in line.strip().split(":")]

        # Get x, y coordinates
        x, y = [int(x) for x in results[1].split(",")]

        # Get angle. The range 0-360 is encoded in 32 increments of 11.25 degrees
        # so 0 means 0 degrees, 31 means 31*11.25 = 348.75 degrees
        # to get degrees, divide by 32, multiply by 360
        angle_encoded = int(results[2])

        # Angle starts at the top (0 degrees) then increases clockwise
        angle_degrees = (angle_encoded / 32) * 360


        # Quality is a float in range [0, 1]
        quality = float(results[3]) * 100

        # Type is ending (1) or bifurcation (2)
        type = 1 if results[4] == "RIG" else 2

        minutiae_list.append((x, y, type, angle_degrees, quality))
    return minutiae_list


def visualize_minutiae(image, minutiae_list, min_quality=0, show_type=False):
    # Parameters for visualization
    line_length = 10
    line_thickness = 2
    point_radius = 3

    for minutia in minutiae_list:
        x, y, type, angle_degrees, quality = minutia

        if quality < min_quality:
            continue

        # MINDTCT outputs an angle that begins at the top, then increases counterclockwise
        # To compute the angle of minutiae lines, we reverse this direction and add 90 degrees
        angle_degrees = ((360 - angle_degrees) + 90) % 360

        # change degrees to radians
        angle_rad = angle_degrees * np.pi / 180
        x_new = int(round(x + line_length * np.cos(angle_rad)))
        y_new = int(round(y - line_length * np.sin(angle_rad)))

        red = [0, 0, 255]
        blue = [255, 0, 0]
        if show_type:
            if type == 1:  # Ending
                color = blue
            elif type == 2:  # Bifurcaton
                color = red
        else:
             color = red

        cv2.circle(image, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.line(image, (x, y), (x_new, y_new), color, line_thickness, lineType=cv2.LINE_AA)
    return image


if __name__ == "__main__":

    enhance_flag = "-b" if CONTRAST_ENHANCEMENT else ""
    minutiae_paths = os.path.join(out_dir, base_name)
    
    # Run binary MINDTCT
    subprocess.call(" ".join([BINARY_PATH, enhance_flag, image_path, minutiae_paths]), shell=True)
    minutiae_list = read_minutiae(minutiae_paths)

    # Vizualize
    image = cv2.imread(image_path)
    image = visualize_minutiae(image, minutiae_list, min_quality=0, show_type=True)

    # Store the image and display
    cv2.imwrite(os.path.join(base_path + '_visualized.png'), image)
    cv2.imshow("image", image)
    cv2.waitKey(0)


