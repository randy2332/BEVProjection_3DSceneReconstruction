# HW1

## Environment:

- OS: Ubuntu 22.04
- Python 3.7

## Execution Instructions

Task 1:

There are photos of the first floor and the second floor respectively.

Please run the following commands in your terminal:

1. Select first floor

```python
python bev.py -f 1

```

1. Select second floor

```python
python bev.py -f 2

```

Task 2:

There are open3D's ICP and my ICP available.

Thus, there are 4 cases:

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

Task1:

1. select first floor
    
    bird’s eye view:
    
    We first select four points in **clockwise** or **counterclockwise** order.
    
    ![image_screenshot_13.10.2023.png](HW1%20dc37153454134aa6b3b831fa28288166/image_screenshot_13.10.2023.png)
    
    front view:
    
    then we can get the projection in front view.
    
    ![Top to front view projection projection.png_screenshot_13.10.2023.png](HW1%20dc37153454134aa6b3b831fa28288166/Top_to_front_view_projection_projection.png_screenshot_13.10.2023.png)
    
2. Select second floor
    
    bird’s eye view:
    
    ![image_screenshot2_13.10.2023.png](HW1%20dc37153454134aa6b3b831fa28288166/image_screenshot2_13.10.2023.png)
    
    front view:
    
    ![Top to front view projection projection2.png_screenshot_13.10.2023.png](HW1%20dc37153454134aa6b3b831fa28288166/Top_to_front_view_projection_projection2.png_screenshot_13.10.2023.png)
    

Task2:

The red line is camera pose and the black line is ground truth.

1. floor 1 with open3D ‘s ICP

![Screenshot from 2023-10-13 23-48-47.png](HW1%20dc37153454134aa6b3b831fa28288166/Screenshot_from_2023-10-13_23-48-47.png)

1. floor 2 with open3D ‘s ICP
    
    ![Screenshot from 2023-10-13 23-51-36.png](HW1%20dc37153454134aa6b3b831fa28288166/Screenshot_from_2023-10-13_23-51-36.png)
    
2. floor 1 with my ICP
    
    ![Screenshot from 2023-10-14 00-01-42.png](HW1%20dc37153454134aa6b3b831fa28288166/Screenshot_from_2023-10-14_00-01-42.png)
    
3. floor 2 with my ICP
    
    ![Screenshot from 2023-10-14 18-16-43.png](HW1%20dc37153454134aa6b3b831fa28288166/Screenshot_from_2023-10-14_18-16-43.png)
