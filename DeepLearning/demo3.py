import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

model1 = load_model('mnist.h5')
# 1.数据加载
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2.数据的处理（归一化）
x_train, x_test = x_train / 255.0, x_test / 255.0
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
loss, acc = model1.evaluate(x_test, y_test, batch_size=128)

print("loss:", loss)
print("accuracy:", acc)

# 8. 模型测试
img = cv2.imread("test.PNG")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.gray()
img = cv2.resize(img, (28, 28)) / 255
img = np.asarray(img, np.float32)

plt.imshow(img)
# print(img)
pred = model1.predict_classes(img.reshape(1, 28, 28))
print(pred)
