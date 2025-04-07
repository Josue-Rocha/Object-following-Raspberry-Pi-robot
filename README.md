# Object-following-Raspberry-Pi-robot

This project will be broken down into five distinct components: Hardware Setup, Object Detection & Tracking, Data Collection & Training, Real-Time Control System, and Integration & Testing. 

## Component 1: Hardware Setup
For starters, I will be using a Raspberry Pi 4 Computer Desktop Kit which has been provided. While this kit contains many crucial components, I will have to find ways to get my hands on a set of actual robot parts. This includes, but is not limited to, wheels, car chassis, motor, motor controllers, wires, a battery, and most obviously...a camera. The design of the physical robot itself might have to be a bit of an afterthought as long at it can perform its basic functions. Afterall, the robot itself is not the point of the project.
## Component 2: Object Detection & Tracking
For this component, I was hoping to implement a deep learning-based detection for object tracking. Since it is still early on in the process, I am still not sure which pre-existing model I should use as the basis for this component. I know certain models will prioritize efficiency over accuracy and vice versa. Potential models include: YOLOv8 Nano, MobileNet SSD, and EfficientDet-Lite.  Additionally, these models can often be very computationally heavy, so I know I may have to use a resource such as TensorFlow Lite to convert it into an optimized format for the Raspberry Pi.
## Component 3: Data Collection & Training
For this component, I was planning on gathering my own custom set of training data. This data would likely have to consist of thousands of images of different colored balls. In order to increase the robustness of the model, I could diversify the data set by taking photos from different angles, distances, with different lighting conditions, and different background. Obviously the data will need to divided into a roughly 70-20-10 split between training, validation, and final testing. I also know that there are resources such as Roboflow and LabelImg which can be used for data annotation to help the model during the training process. As of right now, my plan would be to train the model on the custom dataset using either Google Colab or a local GPU machine.
## Component 4: Real-Time Control System
For this component, I was planning on using PID control to adjust robot movement based on ball position. The basic idea being if the ball moves to the left, then the robot should move to the left. I would have to learn more about using PID controls to make the Raspberry Pi robot move in real time. 
## Component 5: Integration & Testing
Lastly, one of the biggest tasks for me will be to figure out how to connect the object tracking model and the real-time control system. I will have to find a way to take the results of what the tracking model picked up (i.e. it detected that the ball moved to the left) and translate those results into real-time instructions for what the robot should do next (i.e. move to the left). One possible idea could be to consider an algorithm that constantly tries to keep the ball in the center of the robot's "vision" (i.e. center of the screen if we are looking through the camera). In this algorithm, the robot would make any adjustments necessary to reposition itself so that the ball will always remain in the center of the screen. Additionally, it could theoretically be expanded and used as a way for the robot to know when to move forward and backward. Since the ball will effectively be a circle on the screen, the algorthim can try to make sure the circle will always remains the same size. If the circle gets smaller, then the robot would know it has to move forward until it is the right size again. 

## Part 2
For the ball the robot will be tracking, I decided on using a tennis ball. I figured that the bright color, and commonality of tennis balls would make the overall process easier. I have downloaded over 700 JPEG images from the following links:
- https://universe.roboflow.com/rowerup/tennis-ball-8bsra/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- https://universe.roboflow.com/cse-571/tennis-balls-pomv8/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true
- https://universe.roboflow.com/tennis-3ll0a/tennis-ball-icifx/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true
Given the large sum of the images, I have not yet been able to finish sorting the images into training, validation, and "unknown" sets but have been great progress. I have chosen a wide array of images to download with variations in quality, lighting, background, position, etc. so as to have as robust of a data set as possible. Aditionally, I intentionally tried to look for images of tennis balls that are slightly covered (by a person's hand holding it, a dog biting it, etc.). This is because somebody will have to be holding the tennis ball, so it won't appear as a perfect circle in each frame. Given that the goal would be to have the robot track just one ball (and having multiple balls could greatly complicate things) so I filtered the data set so that each image has exactly one tennis ball. No more and no less.

## Part 3
As part of my project to create a tennis ball-tracking robot using the Raspberry Pi 5, I’ve started by building a deep learning-based proof of concept in Google Colab. The goal at this stage is to train a model that can detect a tennis ball from a live webcam feed on a computer. Once this model works reliably in that environment, I’ll adapt and deploy it to the Raspberry Pi to control robot movement in real time.

Firstly, to access my dataset in Google Colab, I mounted my Google Drive. I chose Google Drive because it provides persistent storage across Colab sessions, and it’s easy to organize my dataset into folders for training, validation, and testing. This setup makes working with large numbers of images more manageable, and it’s especially convenient when working across multiple devices or sessions.

Secondly, To handle preprocessing and augmentation, I used TensorFlow's ImageDataGenerator
Here’s a breakdown of what I did and why I did it:

Rescaling Pixel Values
I normalized the pixel values by dividing them by 255 (rescale=1./255). This transforms pixel intensity values from the range [0, 255] to [0, 1]. Neural networks tend to train faster and more effectively when the input data is normalized, so this step was essential.

Resizing
I resized all images to 224x224 pixels. This is the default input size expected by MobileNetV2, the CNN architecture I’m using. It’s also a good balance between preserving visual details and keeping computation lightweight.

Data Augmentation
To improve the model’s robustness and prevent overfitting, I applied real-time data augmentation to the training images. I included:

- Random rotations (up to 30 degrees)

- Shifts in width and height

- Shearing and zooming

- Horizontal flipping

This helps simulate how the tennis ball might appear in different orientations and lighting conditions during real-world use. It also makes the model less reliant on memorizing specific patterns in the training set.

Next, for feature extraction, I chose to use a Convolutional Neural Network (CNN), specifically MobileNetV2, which I imported from TensorFlow’s application module.

Why Use a CNN Instead of Traditional Methods?
Originally, I considered using traditional feature detection methods like color segmentation or contour detection. But those methods can break down under varying lighting conditions, complex backgrounds, or partial occlusion. CNNs, on the other hand, learn hierarchical features directly from the data, making them much more robust and accurate in real-world scenarios.

Why MobileNetV2?
I picked MobileNetV2 for several reasons:

It’s designed for mobile and embedded devices, so it’s lightweight and efficient—perfect for later deployment on the Raspberry Pi.

It’s pre-trained on ImageNet, so I can use it for transfer learning. This lets me benefit from learned visual features without needing massive amounts of training data.

It has fast inference times, which will be useful when the robot needs to react in real time.

Model Setup
I used the MobileNetV2 base with include_top=False, which removes the final classification layer. This allows me to add my own output layer for detecting tennis balls.

I also froze the base model’s weights by setting base_model.trainable = False. This prevents the pre-trained layers from updating during training and makes the training process faster and less data-hungry.

Custom Head
I added a lightweight classification head on top of the MobileNetV2 base:

A GlobalAveragePooling2D layer to flatten the feature maps.

A dense layer with 64 units and ReLU activation for non-linearity.

A final dense layer with 1 unit and a sigmoid activation for binary classification.

