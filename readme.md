# HandWritten Document OCR


## Introduction

In this repository I've used the <a href='https://www.nist.gov/srd/nist-special-database-19' target='_blank'>NIST Special Database 19</a> and Tensorflow for training and testing the convolution neural networks, which recognizes handwritten digits, uppercase letter and lowercase letters. In total 62, classes can be recognized. I am able to get higher accuracy than having all the classes in one neural networks.

## Preprocessing the Training Data
I've downloaded the database by_class.zip from Nist, which are already pre-processed and cleaned for the end-user. So, I've used Image Augmentation for feeding the data to networks. Keras provide a ImageDataGenerator class and are compatible with thier networks. When creating the object ImageDataGenerator, we can provide it with rescaling, shear range, zoom range parameters for more augmented data. Using flow_from_direcctory function of the object with directory_path, target_size,class_mode, batch_size etc.., and in turn it return a iterator-like object into memory which can passed to tensorflow models.  
For better understanding look at incep_trainer.ipynb in the repository.

## Aproach for Recognizing Images

I've used 3 different models to recognize the handwritten text for uppercase, lowercase and numbers respectively and a Random Forest to determine a given image as a number or character.   
For model particulars, I've used InceptionV3 architecture which focus on widening the image, so I stayed with (128,128,1) input image shape. For both uppercase and lowercase, individual models almost acquired similar accuracy of aroung low 93%.  Since classifying number doesn't require complex architectures, I've sticked to AlexNet, for it's robustness and simple architecture, which gave 99.64% of testing accuracy.  
As for the Random Forest, it consists of 170 classifiers and 8 parallel jobs for quick classification, obtained a score of around 90%.  

Check this <a href="https://paperswithcode.com/method/inception-v3#:~:text=Inception%2Dv3%20is%20a%20convolutional,use%20of%20batch%20normalization%20for" target="_blank">paper</a> for better understanding of InceptionV3 Architecture.

## Training

Obviously training a neural network requires a lot of computational power, so i opted to use kaggle.com for training my networks. I've trained my networks for more over 64 with Adam optimizer and 'categorical_accuracy' as a metric. I've used Dropout layers, which help's in not to overfit the network,  with probability of around 0.3, and I didn't experiment with dropout probability. After training I've saved the class_indices from iterator-like object and serialize it into JSON file.

## OCR Application

Using opencv an image is placed into memory, provided a path for image. I've used contours to find the bounding edges of the document(paper) with the help of transformation matrices, if that wasn't successful, skew correction of the documnet is done using projection profile method. For better contour detection image is resized and processed for sharpening. We find contours using opencv and sort them top-to-bottom.  
Deserialized the Neural networks and Random Forest into memory and perfomed classification using custom voting method. Annotate the results onto a copy of image.  
Using ReportLab PDF canvas, drawn the classified letter or numbers onto canvas and special care is taken to draw in correlation to provided image such as similar font-size and letter allignment. The saved PDF is also parsed into a word file, where any misclassification can be corrected. All the results are saved into a unique folder.

## Dependencies
*Tensorflow-gpu (if not Tensorflow should work with minor changes)  
*Sklearn  
*Opencv  
*ProgressBar  
*ReportLab  
*pdf2docx  
