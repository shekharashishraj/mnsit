import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained MNIST model
model = keras.models.load_model('/Desktop/MNIST/mnist_model.h5')

# Initialize the webcam
# Change the value to 0 if it doesn't work for you i.e cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(1)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype('float32') / 255.0
    input_image = np.expand_dims(gray, axis=-1)

    # Make the prediction
    prediction = model.predict(np.expand_dims(input_image, axis=0))
    predicted_class = np.argmax(prediction)

    # Display the prediction
    cv2.putText(frame, str(predicted_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Handwritten Digit Recognition', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
