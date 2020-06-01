#importing neccesery libraries
import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

"""Function for reading the images and reshaping them"""
def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model") # Loading our model that was saved in the training part

prediction = model.predict([prepare('C:/Users/91811/Downloads/download.jpg')]) # predicting the object with our model
a = cv2.imread('C:/Users/91811/Downloads/download.jpg')
plt.imshow(a) #showing the image
plt.show()
#print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])]) # The category of the object
