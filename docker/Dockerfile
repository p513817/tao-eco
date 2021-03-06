##############################################
# 底層環境是吃 tensorrt:20.11-py3
FROM nvcr.io/nvidia/tensorrt:20.11-py3 as tensorrt-dev
COPY ./docker/requirements.sh ./requirements.sh
COPY ./assets/env_tools /tools
# 由於 安裝套件有互動界面會導致報錯 所以先行關閉
ENV DEBIAN_FRONTEND noninteractive
RUN chmod +x ./requirements.sh \
&& ./requirements.sh tensorrt-dev  \
&& rm ./requirements.sh \
&& echo "alias check-env=/tools/check_env_tool " >> ~/.bashrc
ENV PATH="$PATH:/usr/games"
ENV export LC_ALL=C
CMD /tools/hello eco && /bin/bash

##############################################
FROM tensorrt-dev AS deepstream-base
COPY ./docker/requirements.sh ./requirements.sh
ENV DEBIAN_FRONTEND noninteractive
RUN chmod +x ./requirements.sh \
&& ./requirements.sh deepstream-base \
&& rm ./requirements.sh

#############################################
# 由於 deep stream 會吃外部環境，而底層使用了 tensorrt 它就會吃那層的環境
FROM nvcr.io/nvidia/deepstream:5.1-21.02-devel AS deepstream-dev
WORKDIR /workspace
COPY ./docker/requirements.sh ./requirements.sh
COPY ./assets/env_tools /tools
ENV DEBIAN_FRONTEND noninteractive
# 修改 Dockerfile 的底層
SHELL ["/bin/bash", "-c"]
RUN chmod +x ./requirements.sh \
&& ./requirements.sh deepstream-dev \
&& rm ./requirements.sh \
&& echo "alias check-env=/tools/check_env_tool " >> ~/.bashrc 
ENV PATH="$PATH:/usr/games"
ENV export LC_ALL=C
CMD /tools/hello deepstream && /bin/bash
