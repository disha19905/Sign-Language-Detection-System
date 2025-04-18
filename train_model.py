import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# ✅ Set your actual dataset path
DATASET_PATH = "C:/Users/DELL/sign-language-detector/SignLanguageDataset/mini_asl_alphabet"

# Image size and batch size
IMG_SIZE = 64
BATCH_SIZE = 32

# Load and preprocess data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# Save the model
model.save("asl_model.h5")
print("Model saved as asl_model.h5")

# ----------------------------- WEB CAM PREDICTION -----------------------------
# Load the trained model
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
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Normalize the image
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to match model input shape (batch size, height, width, channels)
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Make predictions
    prediction = model.predict(input_frame)

    # Get the label with the highest probability
    predicted_label = labels[np.argmax(prediction)]

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
