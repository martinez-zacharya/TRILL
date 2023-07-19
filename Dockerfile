FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /TRILL 
COPY . /TRILL 
RUN apt-get update && apt-get install -y gcc build-essential
RUN pip install pyg-lib cython torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
RUN pip install trill-proteins 

ENTRYPOINT ["trill"]