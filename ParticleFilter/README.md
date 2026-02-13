# Surgical Robotic Tool Tracking #

Python version of Particle Filter which tracks surgical tools in the following papers:
* Robot Tool Tracking under Partial Visibility: https://arxiv.org/abs/2102.06235
* Surgical Perception (SuPer) Framework: https://arxiv.org/abs/1909.05405
* SuPer Deep: https://arxiv.org/abs/2003.03472
* KeyPoint Optimization: https://arxiv.org/pdf/2010.08054.pdf


## Install Instructions: ##

1. Install ros: http://wiki.ros.org/ROS/Installation
2. Clone repository
3. Create new conda environment with all the required python dependencies:
    ```
    cd <root-directory>/surgical_tool_tracking/  
    conda env create -f environment.yml
    ```

## Run Demo code: ##

Download "journal_dataset" dataset and move the bag and two calibration files to: <root-directory>/surgical_tool_tracking/journal_dataset/

https://drive.google.com/drive/folders/1XMun9NSJ0R2lA0zg6lZNorBaYZ2QAC41?usp=sharing

### Jupter Notebook ###

This demo reads directly from the rosbag dataset so simply open and run the Jupyter-Notebook file: dev_test.ipynb

### ROS Node ###

1. Open a new terminal and run `roscore`
2. Open a new terminal and run:
```
cd <root-directory>/surgical_tool_tracking/painted_markers/
conda activate <conda_env>
python ros_tracking
```
3. Open a new terminal and run:
```
cd <root-directory>/surgical_tool_tracking/journal_dataset/
rosbag play *.bag -r 0.05
```
    
## Citation: ##
```
@ARTICLE{9565398,
  author={Richter, Florian and Lu, Jingpei and Orosco, Ryan K. and Yip, Michael C.},
  journal={IEEE Transactions on Robotics}, 
  title={Robotic Tool Tracking Under Partially Visible Kinematic Chain: A Unified Approach}, 
  year={2021},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TRO.2021.3111441}} 
```
