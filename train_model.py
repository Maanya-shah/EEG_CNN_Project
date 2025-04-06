import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Define path to the training dataset
train_dir = 'C:\\Users\\Maany\\OneDrive\\Desktop\\Test\\eeg_dataset\\train'  # Ensure this path is correct

# Check if the directory exists
print("Train directory exists:", os.path.exists(train_dir))

# Step 1: Load and Preprocess the Dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='sparse',  # Use sparse labels for multi-class classification
    shuffle=True
)

# Step 2: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
   
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
   
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
   
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes: Delta, Theta, Alpha, Beta, Gamma
])

# Step 3: Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
early_stopping = EarlyStopping(monitor='loss', patience=3)  # Stop if loss doesn't improve

history = model.fit(
    train_generator,
    epochs=50,
    callbacks=[early_stopping]
)

# Step 5: Save the Model
model.save('eeg_wave_classifier.h5')

# Step 6: Optionally, Evaluate the Model on the Training Data (if needed)
# We don't have validation data here, but if you'd like to check performance on the training data:
train_loss, train_acc = model.evaluate(train_generator)
print(f'Training accuracy: {train_acc * 100:.2f}%')
