# ME33

## **Deep Reinforcement Learning for Robot Self-Navigation and Room Evacuation**

### *University of Connecticut Senior Design Project*

---

## **Contents**

*1.* [**How to use this code**](#1-how-to-use-this-code)

*1.1* [Installation](#11-installation)  
*1.2* [Launch a simulation in Gazebo](#12-launch-a-simulation-in-gazebo)  
*1.3* [Control the `basicjetbot` model](#13-control-the-basicjetbot-model)

*2.* [**Documentation**](#2-documentation)

*2.1* [`python_gazebo` documentation](#21-python_gazebo-documentation)

---

## *1.* **How to use this code**

### *1.1* **Installation**

---

1. **This project was designed to be used with Ubuntu 18.04**

    We're running an Ubuntu 18.04.5 virtual machine using VMware. 
    (Ubuntu disk image: https://releases.ubuntu.com/18.04/)

2. **Install ROS and Gazebo**

    Follow these instructions for the ROS Desktop-Full Install (includes Gazebo): http://wiki.ros.org/melodic/Installation/Ubuntu

3. **Download and Compile**

    In a new terminal, enter:

    ```
    sudo apt-get install -y python3-catkin-pkg-modules

    sudo apt-get install -y python3-rospkg-modules

    git clone https://github.com/paulblum/ME33.git

    cd ME33

    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
    ```

---

### *1.2* **Launch a simulation in Gazebo**

---

1. **Activate ME33**

    Before you can launch simulations in any new terminal window, you'll need to activate the ME33 workspace:

    ```
    source ME33/devel/setup.bash
    ```

2. **Launch a simulation**

    There are three Gazebo simulation worlds available in ME33, each can be launched using its `roslaunch` command:

    ```
    roslaunch gazebo_sim empty_room.launch

    roslaunch gazebo_sim gradient_room.launch

    roslaunch gazebo_sim obstacle_avoidance.launch
    ```

    *Note:* If you're running on a virtual machine, you may get a GPU-related error when running `roslaunch` for the first time. To fix this, enter the following commands and then try again:

    ```
    echo "export SVGA_VGPU10=0" >> ~/.profile

    export SVGA_VGPU10=0
    ```

3. **Quit a simulation**

    You can close and quit a Gazebo simulation by typing <kbd>ctrl</kbd> + <kbd>C</kbd> in the terminal where it's running.

---

 ### *1.3* **Control the `basicjetbot` model**

---

`basicjetbot` can be controlled using a Python interface in the `python_gazebo` script (`ME33/scripts/python_gazebo.py`).

You can utilize this script to command `basicjetbot` interactively, using a Python shell like `ipython`. To do this, enter the following in a new terminal:

```
cd ME33/scripts

ipython3
```

In the `ipython` shell, import the interface and then initialize a `basicjetbot` controller object:

```python
from python_gazebo import *

jetbot = PythonGazebo()
```

`basicjetbot` can now be controlled in a Gazebo simulation! Try these commands:

```python
jetbot.rotate_to(2)

jetbot.move_to(1,1)

jetbot.teleport_to(0,0,0)
```

You can exit `ipython` by entering "`quit`" in the shell.

---

## *2.* **Documentation**

### *2.1* **`python_gazebo`** documentation

---

***class* `PythonGazebo`:**  
A collection of services for controlling a Gazebo robot via Python.

*Callable functions:*

- **`PythonGazebo(model_name="basicjetbot", ros_rate=10)`:**  
    The constructor.

    *Parameters:*
    - **[str]** `model_name`: The spawned Gazebo model to which this controller should be attached
    - **[int]** `ros_rate`: Rate of issue for commands sent from this controller (Hz)

    *Returns:* a **`PythonGazebo`** object.

    ---

- **`stop()`:**  
    Command the robot to stop moving. Returns the robot's final positional state.

    *Parameters:* None.

    *Returns:*
    - **[float]** final x-coordinate (meters)
    - **[float]** final y-coordinate (meters)
    - **[float]** final heading (radians)

    ---

- **`rotate_to(heading, tol=0.04, max_speed=4, kP=10)`:**  
    Command the robot to rotate itself to some absolute heading, then stop. Returns the robot's final positional state and a collision flag.

    *Parameters:*
    - **[float]** `heading`: target heading (radians, absolute)
    - **[float]** `tol`: final heading tolerance (radians)
    - **[float]** `max_speed`: maximum rotational speed (radians/second)
    - **[float]** `kP`: proportional gain constant

    *Returns:*
    - **[tuple]** positional state
        - **[float]** final x-coordinate (meters)
        - **[float]** final y-coordinate (meters)
        - **[float]** final heading (radians)
    - **[bool]** collision detected? (T/F)

    ---

- **`move_to(x, y, tol=0.04, max_speed=0.6, kP_lin=1.5, kP_rot=7.0)`:**  
    Command the robot to move itself to a set of absolute x-y coordinates, then stop. Returns the robot's final positional state and a collision flag.

    *Parameters:*
    - **[float]** `x`: target x-coordinate (meters, absolute)
    - **[float]** `y`: target y-coordinate (meters, absolute)
    - **[float]** `tol`: final position tolerance (meters)
    - **[float]** `max_speed`: maximum linear speed (meters/second)
    - **[float]** `kP_lin`: proportional gain constant for linear movement
    - **[float]** `kP_rot`: proportional gain constant for rotational corrections

    *Returns:*
    - **[tuple]** positional state
        - **[float]** final x-coordinate (meters)
        - **[float]** final y-coordinate (meters)
        - **[float]** final heading (radians)
    - **[bool]** collision detected? (T/F)

    ---

- **`teleport_to(x, y, heading)`:**  
    Instantly move the robot to a specified poisition. Returns the robot's final positional state.

    *Note:* The robot will not accurately teleport into a collision state. Error greater than 0.01 between target and final coordinates typically indicates that a collision occurred at the target.

    *Parameters:*
    - **[float]** `x`: target x-coordinate (meters, absolute)
    - **[float]** `y`: target y-coordinate (meters, absolute)
    - **[float]** `heading`: target heading (radians, absolute)

    *Returns:*
    - **[float]** final x-coordinate (meters)
    - **[float]** final y-coordinate (meters)
    - **[float]** final heading (radians)

    ---

- **`get_state()`:**  
    Returns the robot's current positional state.

    *Parameters:* None.

    *Returns:*
    - **[float]** current x-coordinate (meters)
    - **[float]** current y-coordinate (meters)
    - **[float]** current heading (radians)

    ---

- **`get_raw_image()`:**  
    Returns the robot's most recently transmitted image data as a list.

    *Parameters:* None.

    *Returns:*
    - **[list]** raw rgb image data

---

***constant* `FOOT`:** 0.3048 meters (Default units in Gazebo are meters & radians).

---

***function* `normalize(radians)`:**  
Map an absolute radian angle to its equivalent value in the interval [-pi, pi].

*Parameters:*
- **[float]** `radians`: any angle (radians)

*Returns:*
- **[float]** normalized angle (radians)

---

***function* `euler_to_quaternion(roll, pitch, yaw)`:**  
Convert a set of Euler angles (roll, pitch, yaw) to a unit quaternion (x, y, z, w). [Learn more about this conversion.](https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_to_quaternion_conversion)

*Parameters:*
- **[float]** `roll`: roll (radians)
- **[float]** `pitch`: pitch (radians)
- **[float]** `yaw`: yaw (radians)

*Returns:*
- **[float]** quaternion x-component
- **[float]** quaternion y-component
- **[float]** quaternion z-component
- **[float]** quaternion w-component

---

***function* `quaternion_to_euler(x, y, z, w)`:**  
Convert a unit quaternion (x, y, z, w) to a set of Euler angles (roll, pitch, yaw). [Learn more about this conversion.](https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion)

*Parameters:*
- **[float]** `x`: quaternion x-component
- **[float]** `y`: quaternion y-component
- **[float]** `z`: quaternion z-component
- **[float]** `w`: quaternion w-component

*Returns:*
- **[float]** roll (radians)
- **[float]** pitch (radians)
- **[float]** yaw (radians)
