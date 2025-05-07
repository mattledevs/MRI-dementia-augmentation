# Dimentia MRI Classification

## Saved Models 
Since saved models are large .keras files, download them from this dropbox link: https://www.dropbox.com/scl/fo/ce9qt8vwjzs18jhe9zm5l/AIi8cop4l1eb53B3kpFWwCo?rlkey=emch6zix4bvf8gpjap0xhk64m&dl=0
Be sure to add the folder 'saved_models' in the root folder and then add the folder name to your gitignore file. 

## Data
https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak

## Model Settings and F1 Documentation

### dimentia_model_VGG16_V1.keras // F1 Score - 0.51
>     # Store ImageDataGenerator parameters in a dictionary
>     imageDataGenerator_params = {
>         'rescale': 1./255,
>     }
>     
>     # Build a Transfer Learning Model
>     base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
>     base_model.trainable = False  # Freeze base
>     
>     # Build the custom classifier
>     model = Sequential([
>         base_model,
>         GlobalAveragePooling2D(), # Reduces output to 1D vector
>         Dense(256, activation='relu'), # Fully connected layer with ReLU
>         Dropout(0.5), # Randomly turn off 50% of neurons to prevent overfitting
>         Dense(4, activation='softmax')  # Final output layer for 4 classes
>     ])
>     
>     # Compile the model
>     model.compile(optimizer=Adam(learning_rate=1e-4),
>                   loss='categorical_crossentropy',
>                   metrics=['accuracy'])

### dimentia_model_VGG16_V2.keras // F1 Score - 0.51

>     # Build a Transfer Learning Model
>     base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
>     base_model.trainable = False  # Freeze base
>     
>     # Build the custom classifier
>     model = Sequential([
>         base_model,
>         GlobalAveragePooling2D(), # Reduces output to 1D vector
>         Dense(256, activation='relu'), # Fully connected layer with ReLU
>         Dense(512, activation='relu'), # 2nd connected layer with ReLU
>         Dropout(0.5), # Randomly turn off 50% of neurons to prevent overfitting
>         Dense(4, activation='softmax')  # Final output layer for 4 classes
>     ])
>     
>     # Compile the model
>     model.compile(optimizer=Adam(learning_rate=1e-4),
>                   loss='categorical_crossentropy',
>                   metrics=['accuracy'])
