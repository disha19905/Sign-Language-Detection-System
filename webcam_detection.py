from collections import deque
import time
import cv2
import tensorflow as tf
import numpy as np
import pyttsx3  # Importing pyttsx3 for Text-to-Speech

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# This sets up the buffer for the last 20 detected letters
buffer = deque(maxlen=20)
last_prediction = ''
last_time = time.time()
cooldown = 1.0  # seconds between same letters

# Load the trained model (make sure the model file is in the current directory)
model = tf.keras.models.load_model("asl_model.h5")

# Define the labels (A to Z)
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to match the model input size
    resized_frame = cv2.resize(frame, (64, 64))

    # Normalize the image
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to match model input shape (batch size, height, width, channels)
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Make predictions using the model
    prediction = model.predict(input_frame)

    # Get the label with the highest probability
    predicted_label = labels[np.argmax(prediction)]

    # Check if the predicted label is different from the last one and if cooldown is passed
    if predicted_label != last_prediction and time.time() - last_time > cooldown:
        # Speak the detected letter
        engine.say(predicted_label)
        engine.runAndWait()

        # Update the last prediction and time
        last_prediction = predicted_label
        last_time = time.time()

    # Show the predicted label on the webcam frame
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit the webcam view when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
