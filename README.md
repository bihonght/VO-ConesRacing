# VO-ConesRacing
Visual Odometry Cones Racing
This is my graduate project for my Master degree and also for UTS Autonomous Racing Team supported and supervised by Prof. Shoudong Huang. I use data which are images from FSOCO dataset (https://fsoco.github.io/fsoco-dataset/) as input for system and to train the model for perception module. 

System Overview: 

The SLAM solution for Formula Student racing, illustrated in Figure 3, starts with input from a monocular camera, which provides data for both Vision Cone Detection and Visual Odometry. The Vision Cone Detection module identifies cones marking the track, while Visual Odometry estimates the vehicle’s movement, adjusting for scale using detected cones. Together, these modules form the perception system, detailed further in the Perception section.

The system’s SLAM algorithm operates in two phases: Local Map Building and Global Mapping. The Local Map captures immediate surroundings to support quick decision-making, feeding information back to improve odometry accuracy. This local data is then integrated into a comprehensive Global Map of the track. Subsequent sections will detail each component, illustrating how they work together to optimize perception and mapping under the dynamic conditions of autonomous racing.
