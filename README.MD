# 🐕 Dog Breed Classifier

This project was developed as the final assignment for the [Machine Learning and Data Science Bootcamp](https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/) by Zero To Mastery, taught by Andrei Neagoie & Daniel Bourke.

I extended the work by making the deep learning model accessible through an interactive web application built with Streamlit.

An interactive web application created with Streamlit that uses a deep learning model to classify dog breeds from uploaded photos.

## Description

This application allows you to upload an image of a dog and identify its breed using a deep learning model. The app features a user-friendly interface that displays the predicted breed along with a confidence chart of the 10 most probable breeds.

## Features

- 📸 Upload images in JPG, JPEG, PNG formats
- 🐶 Automatic detection of a dog's presence in the image
- 🔍 Breed identification with confidence percentage
- 📊 Graphical visualization of the 10 most probable breeds
- 🖼️ View preprocessed image on request
- 📱 Responsive and intuitive interface

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Streamlit
- NumPy
- Matplotlib
- Altair
- Pandas
- PIL
- TensorFlow Hub

## Technical Details

### Image Preprocessing

The application preprocesses uploaded images to adapt them to the input required by the model:

```python
def prepare_image(image, img_size=224):
    """
    Prepares an image for the model:
    1. Resizes to img_size x img_size
    2. Normalizes
    """
    image = tf.image.resize(image, [img_size, img_size])
    return tf.cast(image, tf.float32) / 255.0
```

### Preprocessing includes:

- Resizing the image to 224x224 pixels (standard size for many computer vision models)
- Normalizing pixel values by dividing by 255 to bring them into the range [0,1]
- Converting to float32 data type for compatibility with TensorFlow

## Dog Detection

Before proceeding with breed classification, the app verifies that the image actually contains a dog:

```python
def is_dog(image, model):
    # Preprocessing for MobileNetV2
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_resized = tf.image.resize(img_array, (224, 224))
    img_expanded = tf.expand_dims(img_resized, 0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_expanded)

    # Get predictions
    predictions = model.predict(img_preprocessed)

    # In ImageNet, classes 151-268 are dog breeds
    dog_class_indices = range(150, 268)
    dog_probability = np.sum(predictions[0][dog_class_indices])

    return dog_probability > 0.5, dog_probability
```

This function uses a pre-trained MobileNetV2 model on ImageNet to check if the image contains a dog, leveraging the fact that classes 151 to 268 in ImageNet correspond to dog breeds.

## Breed Prediction

Once the presence of a dog is confirmed, the app proceeds with breed prediction:

```python
def predict_breed(image, model, unique_breeds):
    # Convert PIL image to tensor
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Prepare the image
    img_processed = prepare_image(img_array)
    # Add batch dimension
    img_batch = tf.expand_dims(img_processed, axis=0)
    # Make prediction
    prediction = model.predict(img_batch)
    # Get label
    breed = get_pred_label(prediction[0], unique_breeds)
    # Get confidence score
    confidence = np.max(prediction[0]) * 100
    return breed, confidence, prediction[0], img_processed.numpy()
```

This function:

1. Converts the PIL image to a NumPy array
2. Prepares the image (resizing and normalization)
3. Adds a batch dimension (required by the model)
4. Makes a prediction using the model
5. Identifies the breed with the highest probability
6. Calculates the prediction confidence
7. Returns the predicted breed, confidence, all probabilities, and the preprocessed image

## Results Visualization

1. Results are displayed in a clear and intuitive way:
2. Sorts prediction probabilities to identify the N most likely breeds
3. Creates a DataFrame with breeds and their confidence values
4. Generates a bar chart using Altair
5. Highlights the breed with the highest confidence in green
6. Adds interactive tooltips for a better user experience

## Multiple Image Handling

The app supports uploading multiple images simultaneously,

## Source Code

The notebook with the model building process is available on GitHub:
[Model Notebook](https://github.com/Pandagan-85/ZTM-Machine-learning/blob/main/end_to_end_dog_vision.ipynb)
[Original Dataset on Kaggle](https://www.kaggle.com/c/dog-breed-identification/data)
