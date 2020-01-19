*# TEAM2005 Dockerfile build & run instructions

## Building

To build our source code, make sur you are in the repository containing this README file alongside of the launch folder, the src folder and the Dockerfile file.

Then just call:
```$ docker build -t team2005 .```

## Running the container

Please note that we want to connect host display to the display of our docker container so that the windows opened by OpenCV are actually visible in the host computer and do not raise a " *Gtk: cannot open display:* " error.

### Step 1: Adjust permissions of the X server host

For this intent, before running the container, please call the following command:

```$ xhost +local:root ```

This will actually expose the xhost, so it makes your computer vulnerable during the execution of the code. More details about this command are accessible here: 
[wiki.ros.org](wiki.ros.org/docker/Tutorials/GUI)
### Step 2: Run the container
```
$ docker run --rm -it \
    --env="DISPLAY" \
    --env="QT_X_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --network=host -v \ 
    <path_to_catkin_directory>:/catkin_ws/src  \
    --name team2005 team2005 bash
```


### Step 3: Reset back the permissions of the X server host

When you are done with the run of the docker server, and **only when you are done with the execution**, reset the parameters of the xhost server.

```$ xhost -local:root ```

## Running the code

To run the ros code in the container, just execute the following command:

```
$ cd /catkin_ws
$ roslaunch team2005 team2005.launch
```

Then, launch the simulator and use the following parameters:
   - Team name: team2005
   - Server: ws://127.0.0.1:9090/

