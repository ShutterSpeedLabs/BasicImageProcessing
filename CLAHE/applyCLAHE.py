import numpy as np
from skimage import io, exposure, img_as_uint, img_as_ubyte, transform

# Load the 16-bit monochrome image
image = io.imread('/media/parashuram/AutoData2/city/Denmark/Copenhagen/16BitImages/output/1.png', as_gray=True)

# Check if the image is loaded properly
if image is None:
    print("Error: Image not loaded correctly.")
    exit()

# Normalize the image to the range [0, 1]
image_normalized = exposure.rescale_intensity(image, out_range=(0, 1))

# Zoom the image to 2x using bicubic interpolation
image_zoomed = transform.rescale(image_normalized, scale=2, order=3, mode='reflect', anti_aliasing=True)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = exposure.equalize_adapthist(image_zoomed, clip_limit=0.03)

# Convert the 16-bit CLAHE image to 8-bit for RGB conversion
clahe_image_8bit = img_as_ubyte(clahe)

# Create a 24-bit RGB image by stacking the 8-bit image into three channels
clahe_image_rgb = np.stack([clahe_image_8bit] * 3, axis=-1)

# Save the resulting image
io.imsave('zoomed_clahe_24bit_rgb_image.png', clahe_image_rgb)

# Display the result (optional, requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(image_normalized, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(clahe_image_rgb)
plt.title('Zoomed CLAHE RGB Image')
plt.axis('off')

plt.tight_layout()
plt.show()