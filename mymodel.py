import cv2
import numpy as np
import tensorflow as tf

# Load the model from the .h5 file
model = tf.keras.models.load_model("C:/Users/abhin/Desktop/04.h5")

# Open the camera and set the resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Loop through frames from the camera
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction using the model
    prediction = model.predict(img)

    # Print the prediction
    print(prediction)

    # Display the frame with the prediction
    cv2.imshow('frame', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
