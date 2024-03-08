# Minimum docker file for running the project

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# installing basic linux environment with some libraries
RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install -y libglfw3-dev libgles2-mesa-dev git libgl1-mesa-dev libgl1-mesa-glx xvfb

# git cloning the repository with specified branch
WORKDIR /root/.bittensor
RUN git clone --recursive https://github.com/404-Repo/three-gen-subnet --branch v0_1

# setting up the working directory
WORKDIR /root/.bittensor/three-gen-subnet

# Install all python dependecies for the project
RUN python3 -m pip install -r requirements.txt

# copy the script for initializing the virtual display using xvfb server
COPY entrypoint.sh /usr/local/bin/

# make the script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# initializing virtual display
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# set defualt environment to bash
CMD ["/bin/bash"]
