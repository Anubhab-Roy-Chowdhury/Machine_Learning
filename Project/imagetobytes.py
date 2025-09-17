from PIL import Image
import numpy as np

# Load the image and convert to grayscale
image = Image.open(r"Project\four.png").convert("L")  # "L" = grayscale

# Resize if needed (optional)
# image = image.resize((28,28))  # only if you want exact 28x28

# Convert to numpy array
arr = np.array(image)

# Normalize pixels to 0-1
normalized_arr = arr / 256.0

# Convert to a Python list
pixel_list = normalized_arr.flatten().tolist()

print(pixel_list)        # list of floats between 0 and 1
print(len(pixel_list))   # total number of pixels (28*20 = 560)

