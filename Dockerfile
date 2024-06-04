FROM tensorflow/tensorflow:2.5.1-gpu

ARG USER
ARG UID
ARG GID

# we need sudo especially because jupyter needs access to the .local folder
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${USER}" | chpasswd && adduser ${USER} sudo
#RUN apt-get update -y
#RUN apt-get -y install sudo

#RUN pip install tensorflow_hub
RUN pip install notebook tqdm scikit-learn pandas==1.1.5 numpy~=1.19.2 seaborn==0.11.2 tensorflow-gpu==2.5.1 matplotlib aif360 fairlearn

USER ${UID}:${GID}