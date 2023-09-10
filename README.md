# Brain Tumor Classification from MRI Images
This project is an implementation of an artificial intelligence model for classifying brain MRI images based on the presence of tumors. The model has been developed using PyTorch and trained on a dataset containing brain MRI images with tumor annotations.

## Dataset

![Brain Tumor MRI Dataset](https://storage.googleapis.com/kaggle-datasets-images/1608934/2645886/44583c7826d1bdea68598f0eef8e6cfc/dataset-cover.jpg?t=2021-09-25-22-03-08)

The dataset used in this project is the "Brain Tumor MRI Dataset," which is a combination of three different datasets: figshare, SARTAJ dataset, and Br35H. It comprises a total of 7023 human brain MRI images, categorized into four classes: glioma, meningioma, no tumor, and pituitary adenoma. It's important to note that the "no tumor" class images are sourced from the Br35H dataset.

The dataset used in this project was obtained from Kaggle and is available at the following link: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## About Brain Tumors
A brain tumor is an abnormal collection or mass of cells within the brain. Given the rigid structure of the skull that encases the brain, any growth within this confined space can lead to significant issues. Brain tumors can be classified as either cancerous (malignant) or noncancerous (benign). As benign or malignant tumors grow, they can increase intracranial pressure, potentially causing brain damage and posing life-threatening risks.

## The Significance of Brain Tumor Diagnosis
Early detection and accurate classification of brain tumors are critical in the field of medical imaging. These tasks are pivotal in determining the most appropriate treatment options and significantly impact patient outcomes. Reliable and efficient diagnosis is essential to save lives and enhance the quality of patient care.

## Code Explanation

### Data Preparation
The dataset is organized into training and testing directories. Images are loaded and transformed using PyTorch's ImageFolder and DataLoader classes. Data augmentation is applied, including resizing images to a consistent size of 128x128 pixels.

### Model Architecture (TinyVGG)
The neural network model used for classification is based on the TinyVGG architecture. TinyVGG is a simplified version of the popular VGGNet, designed for efficient and effective image classification. It consists of two convolutional blocks, each followed by a ReLU activation function and max-pooling layers. The final classification is performed through a fully connected layer.

TinyVGG's architecture is chosen for its ability to handle image classification tasks effectively while remaining computationally efficient. It strikes a balance between model complexity and performance, making it suitable for the classification of brain MRI images for this personal project.

#### TinyVGG Architecture Details
 * Convolutional Blocks: The model comprises two convolutional blocks, each consisting of multiple convolutional layers with kernel sizes of 3x3 and ReLU activation functions. These blocks are designed to extract essential features from the input images.

 * Max-Pooling Layers: After each convolutional layer, a max-pooling layer with a kernel size of 2x2 is applied to reduce spatial dimensions and retain important information.

 * Fully Connected Layer: The final classification is performed through a fully connected layer, which takes the flattened output from the convolutional blocks and maps it to the number of classes in the dataset.

### Training and Evaluation
The model is trained over 30 epochs using the Adam optimizer and the cross-entropy loss function. Training progress is monitored through loss and accuracy metrics for both the training and testing datasets. The results are visualized using matplotlib.

### Inference and Visualization
The model is used to make predictions on a subset of test samples, and the predicted classes are compared to ground truth labels. Images and predictions are displayed to provide a visual assessment of the model's performance.

#### Author
This project was developed by William PEOC'H as part of a personal computer science project.
