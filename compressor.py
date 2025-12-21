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