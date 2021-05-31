# SIPCC_ICRA2021

### Mengchao Zhang

### 03/21/2021

### mz17@illinois.edu

This package contains code accompanying the paper
"[Semi-Infinite Programming with Complementarity Constraints for Pose Optimization with Pervasive Contact]"
by M. Zhang, K. Hauser, in the International Conference on Robotics and Automation (ICRA), 2021.

<img src="https://user-images.githubusercontent.com/40469112/111937520-e748cc00-8a95-11eb-9a78-2bc40a499835.png" alt="drawing" width="400"/><img src=https://user-images.githubusercontent.com/40469112/111937842-a604ec00-8a96-11eb-844e-e14c8a624886.png alt="drawing" width="400"/>

## File structure

```
├── data                      World, robot, and object files for running the example code
|   └─── ...
├── README.md                 This file
├── graspopt.py               An example program using SIPCC solving grasping planning problem
└── sipcc/                    The core Python module
    ├── geometryopt.py        SIP code for collision-free constraints between geometries, for objects, robot poses, and robot trajectories.
    ├── __init__.py           Tells Python that this is a module
    ├── sip.py                Generic semi-infinite programming code
    ├── sipcc.py              Generic semi-infinite programming with complementarity constraint solving code
    ├── sipcc_problem.py      Generic semi-infinite programming with complementarity constraint problem defination
    ├── graspingopt.py        SIPCC code for grasping planning
    ├── mpcc.py		      MPCC problem solver (currently only has an SNOPT interface, may add our own MPCC solver in the future)
    └── objective.py          Generic objectives for optimization problems
```


## Dependencies

This package requires

1. Numpy/Scipy

2. [OSQP](http://osqp.org) for quadratic program (QP) solving.  OSQP can be
   installed using

> pip install osqp

   Other solvers might be supported in the future.

3. The [Klampt](https://klampt.org) 0.8.x Python API (https://klampt.org) to be installed.  `pip install klampt` may work.

4. [cvxopt](https://cvxopt.org/). Cvxopt can be installed using 

> pip install cvxopt

5. sklearn can be installed using 

> pip install scikit-learn

6. pyoptsparse: https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html

7. SNOPT: http://www.sbsi-sol-optimize.com/asp/sol_snopt.html
