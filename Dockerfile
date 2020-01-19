FROM dungpb/dira_ros:ros-python

RUN /bin/bash -c 'cd /catkin_ws/src; mkdir team2005'

COPY . /catkin_ws/src/team2005

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /catkin_ws; catkin_make'
