import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test_images/1.png')
imgplot = plt.imshow(img)
print(img)
plt.show()