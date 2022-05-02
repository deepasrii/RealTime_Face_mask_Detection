#                             RealTime_Face_mask_Detection

A Face Mask detector in real-time video streams with OpenCv, Tensorflow/keras in Deep Learning is implemented.

phase 1: train_mask_detector.py

Train a custom deep learning model to detect whether a person is wearing a mask or not. 
The dataset contains 763 images-with mask and 686 images without mask.
The MobileNetV2 classifier fine tunes the input dataset and creates mask_detector.model.
Loading the model, the plot.png file is created based on accuracy and loss curves.

phase 2: mask_detection_webcam.py

Here, videostream function is imported for processing every frame of the webcam stream along with necessary libraries. 
Load the face detector using dnn and face mask classifier model using mobilenetv2.
Now, the face mask detector is capable of running in real-time.
