# NYCU-perception-and-decision-making-in-intelligent-systems-hw1

## Enviroment:

OS : ubuntu 22.04

Python 3.7

---

## Exercute Program

Task1 :

There are photos of the first floor and the second floor respectively.

Please run in your terminal

1. select first floor

```python
python bev.py -f 1
```

1. select second floor

```python
python bev.py -f 2
```

Task2:

There are open3D’s ICP and my ICP respectively.

Thus, there are 4 case:

1. floor 1 with open3D ‘s ICP

```python
python reconstruct.py -f 1 -v open3d
```

1. floor 2 with open3D ‘s ICP

```python
python reconstruct.py -f 2 -v open3d
```

1. floor 1 with my ICP

```python
python reconstruct.py -f 1 -v my_icp
```

1. floor 2 with my ICP

```python
python reconstruct.py -f 2 -v my_icp
```
