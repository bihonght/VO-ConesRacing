# Visual SLAM Driverless
Visual Slam Driverless Racing
This is my graduate project for my Master degree and also for UTS Autonomous Racing Team supported and supervised by Prof. Shoudong Huang. I use data which are images from FSOCO dataset (https://fsoco.github.io/fsoco-dataset/) as input for system and to train the model for perception module. 

A demo:  
<p align = "center">
  <img src = "https://github.com/bihonght/VO-ConesRacing/blob/main/media/demo.gif" height = "240px">
</p>

In the above figure:  
**Right** is a sequence of input images.    
**Left** is the camera trajectory corresponding to the right video: **Blue** line is from VO. Red markers are the detected cone locations.

# Report
My pdf-version course report is [here](
https://github.com/bihonght/VO-ConesRacing/blob/main/media/Capstone%20Report.pdf). It has a more clear decription about the algorithms than this README, so I suggest to read it.

# System Overview:

The SLAM solution for Formula Student racing, illustrated in Figure below, starts with input from a monocular camera, which provides data for both Vision Cone Detection and Visual Odometry. The Vision Cone Detection module identifies cones marking the track, while Visual Odometry estimates the vehicle’s movement, adjusting for scale using detected cones. Together, these modules form the perception system, detailed further in the Perception section. 

The system’s SLAM algorithm operates in two phases: Local Map Building and Global Mapping. The Local Map captures immediate surroundings to support quick decision-making, feeding information back to improve odometry accuracy. This local data is then integrated into a comprehensive Global Map of the track. Subsequent sections will detail each component, illustrating how they work together to optimize perception and mapping under the dynamic conditions of autonomous racing.

<p align = "center">
  <img src = "https://github.com/bihonght/VO-ConesRacing/blob/main/media/architecture%20(1).png" height = "240px">
</p>

Due to a shortage of input dataset, which is a large gap time between two consecutive input images from camera. I decide to use each cone and its size to be landmarks for odemetry and map construction. This is not a final and perfect solution for commercial purpose, however, it can be a new direction for low-cost SLAM system. 

https://youtu.be/kB_CEXcpZJ4?si=Nxg1P3xLAeds83MX
