FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
  procps \
  && apt-get clean

COPY environment.yml .
COPY ./SHMModels /SHMModels
RUN conda env create -f environment.yml
RUN echo "source activate shmpy" > ~/.bashrc
ENV PATH /opt/conda/envs/shmpy/bin:$PATH

COPY ./python /python
RUN chmod +x /python/cli.py
ENV PATH="$PATH:/python"
WORKDIR /python
CMD python -W ignore keras_python_notrain.py
