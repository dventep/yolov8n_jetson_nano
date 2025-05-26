FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3-opencv nano wget git curl \
    && apt-get clean

# Instala Jetson.GPIO solo si es Jetson
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then pip install Jetson.GPIO; fi

# Instalar Ultralytics
RUN pip install ultralytics

# Establecer permisos del grupo gpio
#RUN groupadd -f -g 999 gpio && usermod -a -G gpio root

# Copiar tu modelo y código
COPY Model/ /workspace/Model/
COPY main.py /workspace/

WORKDIR /workspace
CMD ["python3", "main.py"]
