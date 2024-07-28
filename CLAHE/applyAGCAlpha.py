import numpy as np
from skimage import io, exposure, img_as_uint, img_as_ubyte, transform
from scipy import stats

def adaptive_gamma_correction(image, alpha=1.0):
    image_flattened = image.flatten()
    mean = np.mean(image_flattened)
    median = np.median(image_flattened)
    
    # Calculate skewness
    skewness = stats.skew(image_flattened)
    
    # Determine gamma based on skewness
    if skewness < 0:
        gamma = 1 - skewness
    else:
        gamma = 1 / (1 + skewness)
    
    # Apply gamma correction
    corrected = np.power(image, gamma * alpha)
    
    # Normalize to [0, 1]
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min())
    
    return corrected

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

# Apply Adaptive Gamma Correction
agc_image = adaptive_gamma_correction(image_zoomed)

# Convert the AGC image to 8-bit for RGB conversion
agc_image_8bit = img_as_ubyte(agc_image)

# Create a 24-bit RGB image by stacking the 8-bit image into three channels
agc_image_rgb = np.stack([agc_image_8bit] * 3, axis=-1)

# Save the resulting image
io.imsave('zoomed_agc_24bit_rgb_image.png', agc_image_rgb)

# Display the result (optional, requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(image_normalized, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(agc_image_rgb)
plt.title('Zoomed AGC RGB Image')
plt.axis('off')

plt.tight_layout()
plt.show()