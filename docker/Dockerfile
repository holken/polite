ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/


RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install pip install highway-env>=1.7.1 && \
    pip install mlagents==0.28.0 && \
    pip install gym_unity && \
    rm -rf $HOME/.cache/pip

RUN apt-get update \
  && apt-get install -y gcc \
  && apt-get install -y build-essential \
  && apt-get install -y python-dev \
  && apt-get install -y python3-dev \
  && apt-get install -y libosmesa6-dev \
  && apt-get install -y libopengl0 \
  && apt-get install -y libglew-dev \
  && apt-get install -y libc-dev \
  && apt-get install -y libgl1-mesa-glx libglfw3 \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y \
  && apt-get install -y patchelf

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

ENV PATH=$VENV/bin:$PATH
RUN pip install tensorboardX && pip install pymongo && pip install -U gym==0.26.2 && pip install mujoco==2.3.3
COPY docker/entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
