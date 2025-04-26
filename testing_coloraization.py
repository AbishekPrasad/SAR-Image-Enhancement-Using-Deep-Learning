import cv2
import numpy as np
import matplotlib.pyplot as plt  

# Paths to your one SAR L-image and one Optical RGB image
sar_image_path = 'SAR.jpg'      # SAR grayscale image (0-255)
optical_image_path = 'color.jpg'  # Optical RGB image

# 1. Load the SAR image (grayscale)
sar_gray = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
sar_L = sar_gray.astype("float32") / 255.0 * 100.0           # Normalize to [0,100]

# 2. Load the Optical RGB image
optical_bgr = cv2.imread(optical_image_path)  # Load as BGR
optical_rgb = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
optical_rgb_float = optical_rgb.astype("float32") / 255.0  # Normalize to [0,1]

# 3. Convert Optical RGB to LAB
optical_lab = cv2.cvtColor(optical_rgb_float, cv2.COLOR_RGB2LAB)

# 4. Extract AB channels from optical image
_, a_channel, b_channel = cv2.split(optical_lab)
ab_channels = np.stack((a_channel, b_channel), axis=-1)  # shape: (H, W, 2)

# 5. Merge SAR L with Optical AB to form LAB
sar_L_expanded = sar_L[:, :, np.newaxis]  # Expand dims to (H, W, 1)
fake_lab = np.concatenate((sar_L_expanded, ab_channels), axis=2)  # shape: (H, W, 3)

# 6. Convert LAB back to RGB
fake_lab = fake_lab.astype("float32")
reconstructed_rgb_float = cv2.cvtColor(fake_lab, cv2.COLOR_LAB2RGB)

# 7. Clip values and convert to 8-bit
reconstructed_rgb_float = np.clip(reconstructed_rgb_float, 0, 1)
reconstructed_rgb = (reconstructed_rgb_float * 255).astype("uint8")

# 8. Display images using matplotlib
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Optical Image (RGB)")
plt.imshow(optical_rgb)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstructed Image (SAR L + Optical AB)")
plt.imshow(reconstructed_rgb)
plt.axis('off')

plt.show()
