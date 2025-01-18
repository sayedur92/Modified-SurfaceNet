# Modified-SurfaceNet

This project explores the development and evaluation of BRDF rendering using deep learning models. The model training and inference are provided as separate scripts, which can be run to generate results.


Training the Model:
To train the model, simply run the train.py script. The script will handle the process of training the model using the provided dataset and configurations.

Running Inference
After training the model, you can use the inference.py script for generating predictions or conducting inference with the trained model.



Ensure that you have the dataset ready and accessible before starting the training process.


pip install -r requirements.txt
File Structure
train.py: Script for training the model.
inference.py: Script for running inference with the trained model.
requirements.txt: List of dependencies for the project.
dataset/: Directory to store the dataset for training.
results/: Directory to store the output of inference and other results.
Evaluation Metrics
The performance of the models is evaluated using the following metrics:

VGG Loss: Measures the perceptual similarity between the rendered image and the input image.
Sketch Loss: Ensures the fine details of the generated reflectance maps align with the input sketch.
Reflectance Map Accuracy: Compares the predicted reflectance maps to the ground-truth maps.
You can modify the scripts to include more metrics or experiment with different loss functions based on your requirements.

Contact Information
For any issues or inquiries, feel free to reach out to the project maintainer:

Name: Sayedur Rahman
Email: [sayedur318@gmail.com]
