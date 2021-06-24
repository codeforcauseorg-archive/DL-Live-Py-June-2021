import cv2
import numpy as np
from tensorflow.keras.models import load_model

img = cv2.imread("item.png")
model = load_model("mnist_model")

out = img.mean(axis=2)

output = model.predict(np.array([out]))
print(output.argmax(axis=1))


# cv2.imshow("A number", img)
#
# cv2.waitKey(0)




