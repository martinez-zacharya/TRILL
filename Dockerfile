FROM ubuntu:22.04

WORKDIR /TRILL 
COPY . /TRILL 
RUN apt-get update && apt-get install -y gcc build-essential pip python3 
RUN pip install transformers pyg-lib pydantic==1.8.2 cython torch==1.13.1 torch-scatter torch-sparse trill-proteins torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
RUN pip install accelerate==0.20.3 

ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["trill"]