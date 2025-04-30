FROM r-base:4.0.5

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install htop -y

RUN apt-get install git -y

RUN apt-get install tmux -y

# install python
RUN apt-get install -y --no-install-recommends build-essential libpq-dev python3.8 python3-pip python3-setuptools python3-dev
WORKDIR /tmp/

# RUN python3 -m venv .venv
# RUN .venv/bin/activate

RUN pip3 install --break-system-packages --upgrade pip

COPY ./requirements_r_image.txt .

VOLUME ["/tokens"]

RUN pip install --break-system-packages -r requirements_r_image.txt

RUN rm ./requirements_r_image.txt


ARG USERNAME=ruggeri
ARG USER_UID=2565
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME