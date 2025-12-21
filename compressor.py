import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress(image_path, k):
    # Load the image and convert to grayscale matrix A
    img = Image.open(image_path).convert('L')
    A = np.array(img)

    # Computer Singular Value Decomposition
    # As noted, libraries use bidiagonalization and QR decomposition to optimize
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate to rank k
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Reconstruct the compressed image
    A_k = np.dot(U_k, np.dot(S_k, Vt_k))

    return A, A_k, U_k, S_k, Vt_k

# Test run
original, compressed, Uk, Sk, Vtk = compress('example.jpg', 50)

# Calculate data savings
m, n = original.shape
original_size = m * n
compressed_size = (m * 50) + (50 * 50) + (50 * n)
ratio = compressed_size / original_size
print(f"Compression Ratio: {ratio:.2f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original, cmap='gray')
plt.subplot(1, 2, 2)
plt.title(f"Rank-{50} Compressed Image")
plt.imshow(compressed, cmap='gray')
plt.show()