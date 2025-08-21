# MNIST-Digit-Classification-using-an-Artificial-Neural-Network
This project demonstrates a multi-class classification task to predict handwritten digits using the well-known MNIST dataset. The prediction is accomplished by building and training an Artificial Neural Network (ANN) with the help of the TensorFlow and Keras libraries.


Dataset : 
The project utilizes the MNIST (Modified National Institute of Standards and Technology) database, which is a large collection of handwritten digits. It is a standard dataset for benchmarking image classification algorithms.

• Dataset Size:

 • Training Set: 60,000 images of 28x28 pixels.

 • Test Set: 10,000 images of 28x28 pixels.

• Link to the Dataset: The dataset is conveniently loaded directly through the keras.datasets.mnist.load_data() function within the notebook. More information can be found at the [official MNIST database website.](http://yann.lecun.com/exdb/mnist/)


Dependencies : 
To run this notebook, you will need the following Python libraries installed:

• TensorFlow: An end-to-end open-source platform for machine learning.

• Keras: A high-level neural networks API, running on top of TensorFlow.

• Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.

• Scikit-learn: A tool for data mining and data analysis.



Model Architecture : 
An Artificial Neural Network (ANN) is constructed using a Sequential model from Keras. The architecture is as follows:

1. Flatten Layer: This layer converts the 2D image data (28x28 pixels) into a 1D array of 784 neurons, which serves as the input to the network.

2. Dense Layer 1: A fully connected hidden layer with 128 neurons and a ReLU (Rectified Linear Unit) activation function.

3. Dense Layer 2: Another fully connected hidden layer with 32 neurons and a ReLU activation function.

4. Dense Layer 3 (Output Layer): The final output layer with 10 neurons (one for each digit from 0 to 9) and a Softmax activation function. The softmax function outputs a probability distribution over the 10 classes.


Training : 
The model is compiled and trained with the following parameters:

• Loss Function: sparse_categorical_crossentropy is used, which is suitable for multi-class classification problems where the labels are integers.

• Optimizer: The Adam optimizer is employed to update the network weights during training.

• Metrics: The model's performance is monitored using the accuracy metric.

• Epochs: The model is trained for 25 epochs.

• Validation Split: 20% of the training data is used for validation to monitor the model's performance on unseen data during training.


Results : 
The model's performance is evaluated on the test set, and the training history is visualized to observe the changes in loss and accuracy over epochs.

• Accuracy: The final accuracy on the test set is approximately 98.04%.

• Training History:

 • The loss plot shows the training and validation loss decreasing over epochs, indicating that the model is learning.

 • The accuracy plot shows the training and validation accuracy increasing, which is the expected behavior.

How to Run : 
1. Ensure you have the required dependencies installed.
2. Open the MNIST_Classification.ipynb file in a Jupyter Notebook environment.

3. Run the cells sequentially to load the data, build the model, train it, and evaluate its performance.





