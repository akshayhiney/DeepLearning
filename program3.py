
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize input images (pixel values 0-255 → 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential([
    Flatten(input_shape=(28, 28)),          # Flatten 28x28 image → 784 vector
    Dense(512, activation='relu'),          # Hidden layer 1
    Dropout(0.3),                           # Dropout for regularization
    Dense(256, activation='relu'),          # Hidden layer 2
    Dropout(0.3),
    Dense(128, activation='relu'),          # Hidden layer 3
    Dense(64, activation='relu'),           # Hidden layer 4
    Dense(10, activation='softmax')         # Output layer (10 classes)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    batch_size=128,
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.4f}")


def predict_custom_image(image_path):

    img = Image.open(image_path).convert("L")   # Convert to grayscale
    img = img.resize((28, 28))                  # Resize to 28x28

    
    img_array = np.array(img).astype("float32") / 255.0


    if np.mean(img_array) < 0.5:  
        img_array = 1 - img_array  

    pred_probs = model.predict(img_array.reshape(1, 28, 28))
    pred_label = np.argmax(pred_probs)

   
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Predicted: {pred_label}")
    plt.axis("off")
    plt.show()

    return pred_label


prediction = predict_custom_image("/content/sample_data/5.png")
print("Model Prediction:", prediction)

