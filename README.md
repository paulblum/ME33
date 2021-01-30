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

    roslaunch room_evac_gazebo room_evac.launch
    ```

    *Hint:* If you're running on a virtual machine, you may get a GPU-related error after the `roslaunch` command. Enter the following and then try again:

    ```
    echo "export SVGA_VGPU10=0" >> ~/.profile

    export SVGA_VGPU10=0
    ```

    You can close and quit Gazebo by typing <kbd>ctrl</kbd> + <kbd>C</kbd> in this terminal.

5. **Control the basicjetbot**

    `basicjetbot` is controlled via a Python interface in the GazeboUtils script (`ME33/src/room_evac_gazebo/script/GazeboUtils.py`). 
    
    You can utilize this script to command `basicjetbot` interactively, using a Python shell like `ipython`. To do this, enter the following in a new terminal:

    ```
    cd ME33/src/room_evac_gazebo/script

    ipython3
    ```

    In the Python shell (or in a script), import the GazeboUtils script and then initialize a controller object:

    ```python
    from GazeboUtils import *

    bot = DiffDriveControl()
    ```

    `basicjetbot` can now be controlled in the simulation! Try these commands:
    
    ```python
    bot.rotate_to(2)

    bot.move_forward(3)

    bot.set_state(1,1)

    bot.move_to(2,2)
    ```

    You can exit `ipython` by entering "`quit`" in the shell.
