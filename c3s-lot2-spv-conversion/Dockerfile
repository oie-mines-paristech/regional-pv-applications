FROM continuumio/miniconda3

WORKDIR /src/c3s-lot2-spv-conversion

COPY environment.yml /src/c3s-lot2-spv-conversion/

RUN conda install -c conda-forge gcc python=3.12 \
    && conda env update -n base -f environment.yml

COPY . /src/c3s-lot2-spv-conversion

RUN pip install --no-deps -e .
