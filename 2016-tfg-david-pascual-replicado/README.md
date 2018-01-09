# 2016-tfg-david-pascual

**Project Name:** Deep Learning on RGBD sensors

**Author:** David Pascual Hernández [d.pascualhe@gmail.com]

**Academic Year:** 2016/2017

**Degree:** Degree in Audiovisual Systems and Multimedia Engineering

**Mediawiki:** http://jderobot.org/Dpascual-tfg

**Tags:** Deep Learning, Keras

**State:** Developing 

## Usage
<code>digitclassifier</code> is a JdeRobot component which captures live video and classifies digits in the sequence with a convolutional neural network built with Keras. In order to launch it with Python 2.7 you must install: 
* JdeRobot ([installation guide](http://jderobot.org/Installation))
* OpenCV 3 (it will be automatically installed with JdeRobot)
* Keras ([installation guide](https://keras.io/#installation))

If you want to launch <code>digitclassifier</code>, open a terminal and run:
<pre>
cameraserver cameraserver.cfg
</pre>
This command will start <code>cameraserver</code> driver which will serve video from the webcam. In another terminal, access the location where you've cloned the repo and run:
<pre>
python digitclassifier.py digitclassifier.cfg
</pre>
That command should launch the component and you should see something like this:
![Alt Text](https://media.giphy.com/media/xT0xevE4RgzA4CTEju/giphy.gif)

More info: [http://jderobot.org/Dpascual-tfg]
