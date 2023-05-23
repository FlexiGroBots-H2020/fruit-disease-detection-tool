# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt /wd
RUN pip install -r requirements.txt

COPY Detic/ /wd/Detic/
WORKDIR /wd/Detic
RUN pip install -r requirements.txt

COPY detectron2/ /wd/detectron2/
WORKDIR /wd/detectron2
RUN pip install -e .

WORKDIR /wd
COPY fruit_pipeline.py /wd
COPY fruit_detection.py /wd
COPY fruit_detection_utils.py /wd
COPY fruit_detection_kserve.py /wd
COPY kserve_backbone_fruit.py /wd
COPY kserve_utils.py /wd
COPY models_clip/ /wd/models_clip/
COPY models/ /wd/models/
COPY datasets /wd/datasets/
COPY X_Decoder /wd/X_Decoder/
COPY xdcoder_utils.py /wd
COPY detic_utils.py /wd
COPY yolo_utils.py /wd
COPY yolov8l-cls.pt /wd
COPY yolov8x-seg.pt /wd

RUN chmod -R 777 /wd

USER jovyan
#RUN mkdir -p /wd/outputs/face_detection_cache/FaceDetector/


ENTRYPOINT ["python", "-u", "kserve_backbone_fruit.py"]
