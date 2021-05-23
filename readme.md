# HandWritten Document OCR
<hr>

## Introduction
<hr>
In this repository I've used the <a href='https://www.nist.gov/srd/nist-special-database-19' target='_blank'>NIST Special Database 19</a> and Tensorflow for training and testing the convolution neural networks, which recognizes handwritten digits, uppercase letter and lowercase letters. In total 62, classes can be recognized. I was able gto achieve an accuracy of 92% while testing neural networks.

## Preprocessing the Training Data
<hr>
I've downloaded the database by_class.zip from Nist, which are already pre-processed and cleaned for the end-user. So, I've used Image Augmentation for feeding the data to networks. Keras provide a ImageDataGenerator class and obviously is compatible with thier networks. When creating the object ImageDataGenerator we can provide it with rescaling parameters. Using flow_from_direcctory function of the object, we provide the directory path, target_size,class_mode etc..,   
For better understanding look at incep_trainer.ipynb in the repository.

## Aproach for Recognizing Images
<hr>
I've used 3 different models to recognize the handwritten text for uppercase, lowercase and numbers respectively and an Random Forest to determine a given image as a number or character.   
For model particulars, I've used InceptionV3 architecture which focus on widening the image, so stayed with (128,128,1) input image shape. Since classifying number doesn't require complex architectures, I've sticked to AlexNet. which gave 99.84% of accuracy while training and testing.  
As for the Random Forest, it consists of 170 classifiers and obtained a score of 90%.
Check this <a href="https://paperswithcode.com/method/inception-v3#:~:text=Inception%2Dv3%20is%20a%20convolutional,use%20of%20batch%20normalization%20for" target="_blank">paper</a>for better understanding of InceptionV3 Architecture.

## Training
<hr>
Obviously training a neural network requires a lot of computational power, so i opted to use kaggle.com for training my networks. I've trained my networks for more over 64 with Adam optimizer and 'categorical_accuracy' as a metric. I've used Dropout layers, which help's in not to overfit the network,  with probability of around 0.3, and I didn't experiment with dropout probability. 

## OCR Application
