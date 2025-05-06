import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load our pattern
gray = cv2.imread("pattern.png", cv2.IMREAD_GRAYSCALE)

############################################################################
# Step 1: build the Gabor kernel that will enhance for us vertically oriented patches:

# ***TASK*** Select parameters of your Gabor kernel here:
ksize = 11      # Chosen in the range of 5-15
sigma = 3.0     # Chosen in the range of 2.0-4.0
theta = 0.0     # Keep it 0.0 to focus on vertically-oriented patterns
lbd = 3.0       # Chosen in the range of 2.0-4.0
gamma = 1.0     # Keep it 1.0
psi = 0.0       # Keep it 0.0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lbd, gamma, psi, ktype=cv2.CV_32F)

# Normalize the kernel and remove the DC component
kernel /= kernel.sum()
kernel -= kernel.mean()

# Visualize the kernel
xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, kernel, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
plt.show()

############################################################################
# Step 2: image filtering

res1 = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
cv2.imshow("Filtering result", res1)

############################################################################
# Step 3: image binarization using Otsu's method

th2, res2 = cv2.threshold(res1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's binarization", res2)

############################################################################
# Step 4: morphological operations and getting your area of interest annotated

se_size = 7  # Chosen in the range of 5-15
se = np.ones((se_size, se_size), np.uint8)

# Choosing a morphological operation (Closing to fill gaps)
type = cv2.MORPH_CLOSE  # Options: MORPH_CLOSE, MORPH_OPEN, MORPH_ERODE, MORPH_DILATE
res3 = cv2.morphologyEx(res2, type, kernel=se)

cv2.imshow("Areas with vertical pattern annotated", res3)

cv2.waitKey(0)
cv2.destroyAllWindows()
