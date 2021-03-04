# ME33

## **Deep Reinforcement Learning for Robot Self-Navigation and Room Evacuation**

### *University of Connecticut Senior Design Project*

---

### Run the code:

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

4. **Launch the Simulation in Gazebo**

    In the terminal (still in ME33 directory), enter:

    ```
    source devel/setup.bash

    roslaunch gazebo_sim empty_room.launch
    ```

    *Hint:* If you're running on a virtual machine, you may get a GPU-related error after the `roslaunch` command. Enter the following and then try again:

    ```
    echo "export SVGA_VGPU10=0" >> ~/.profile

    export SVGA_VGPU10=0
    ```

    You can close and quit Gazebo by typing <kbd>ctrl</kbd> + <kbd>C</kbd> in this terminal.

5. **Control the basicjetbot in Python**

    `basicjetbot` can be controlled using a Python interface in the `python_gazebo` script (`ME33/scripts/python_gazebo.py`).

    You can utilize this script to command `basicjetbot` interactively, using a Python shell like `ipython`. To do this, enter the following in a new terminal:

    ```
    cd ME33/scripts

    ipython3
    ```

    In the Python shell (or in a script), import `python_gazebo` and then initialize a controller object:

    ```python
    from python_gazebo import *

    bot = PythonGazebo()
    ```

    `basicjetbot` can now be controlled in the simulation! Try these commands:

    ```python
    bot.rotate_to(2)

    bot.move_to(1,1)

    bot.teleport_to(0,0,0)
    ```

    You can exit `ipython` by entering "`quit`" in the shell.

---

### Full documentation for the `python_gazebo` interface:

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

---
