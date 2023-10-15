# NYCU-perception-and-decision-making-in-intelligent-systems-hw1


## Data Preparation

To get started, download the replica dataset from the following links:

1. [Link 1](https://drive.google.com/file/d/1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4/view?usp=sharing)
2. [Link 2](https://github.com/facebookresearch/Replica-Dataset)

## Environment:

- OS: Ubuntu 22.04
- Python 3.7

## Execution Instructions

First, run [load.py](http://load.py/) to collect your data.

For the first floor, use the following command in your terminal:

```python
python load.py -f 1

```

For the second floor, use the following command in your terminal:

```python
python load.py -f 2

```

Task 1:

There are photos available for the first floor and the second floor.

To select the first floor, use the following command:

```python
python bev.py -f 1

```

To select the second floor, use the following command:

```python
python bev.py -f 2

```

Task 2:

There are two options available for the ICP (Iterative Closest Point) algorithm: open3D's ICP and my ICP.

There are 4 cases to consider:

1. Floor 1 with open3D's ICP

```python
python reconstruct.py -f 1 -v open3d

```

1. Floor 2 with open3D's ICP

```python
python reconstruct.py -f 2 -v open3d

```

1. Floor 1 with my ICP

```python
python reconstruct.py -f 1 -v my_icp

```

1. Floor 2 with my ICP

```python
python reconstruct.py -f 2 -v my_icp

```

## Results

Task 1:

1. Selection of the first floor
    - Bird's eye view:
    
    We first select four points in either **clockwise** or **counterclockwise** order.
    
    ![bevfloor1.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/bevfloor1.png)
    
    - Front view:
    
    Then, we can obtain the projection in the front view.
    
    ![frontfloor1.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/frontfloor1.png)
    
2. Selection of the second floor
    - Bird's eye view:
    
    ![bevfloor2.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/bevfloor2.png)
    
    - Front view:
    
    ![forntfloor2.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/forntfloor2.png)
    

Task 2:

The red line represents the camera pose, and the black line represents the ground truth.

1. Floor 1 with open3D's ICP
    
    ![open3dfloor1.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/open3dfloor1.png)
    
2. Floor 2 with open3D's ICP
    
    ![open3dfloor2.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/open3dfloor2.png)
    
3. Floor 1 with my ICP
    
    ![myicpfloor1.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/myicpfloor1.png)
    
4. Floor 2 with my ICP
    
    ![myicpfloor2.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw1/blob/main/myicpfloor2.png)
