FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    gdal-bin \
    proj-bin \
    libproj-dev \
    libspatialindex-dev \
    libgeos-dev \
    libgeos++-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.2

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    gdal==${GDAL_VERSION} \
    geopandas==0.14.4 \
    rasterio==1.3.10 \
    shapely==2.0.4 \
    fiona==1.9.6 \
    pyproj==3.6.1 \
    rtree==1.2.0 \
    pandas==2.2.2 \
    matplotlib==3.8.5

RUN pip install --no-cache-dir \
    folium==0.17.0 \
    mapclassify==2.6.1

WORKDIR /home/data

RUN useradd -m -s /bin/bash geosandbox && \
    chown -R geosandbox:geosandbox /home/data

RUN python3 -c "import geopandas; import rasterio; import fiona; print('All GIS dependencies loaded successfully')"
