# CarND-Traffic-Light-Classifier
An implementation of Traffic Light Classifier for the Udacity Self Driving Car Capstone Project.

This traffic light classifier was trained using the [SSD Inception V2 model](https://arxiv.org/pdf/1512.00567v3.pdf). 

The Base Models used here are taken from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)



# Set up Instructions

## Linux

1.  Install Tensorflow 1.4

    pip install tensorflow==1.4
    
2.  Install Other packages

    pip install matplotlib

    sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

    sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

    pip install tensorflow-gpu==1.4

    pip install matplotlib


3.  Create a new directory somewhere and name it tensorflow. Clone the Tensorflow's models repo by executing

    git clone https://github.com/tensorflow/models.git

4.  Checkout the version that works with Tensorflow 1.4. Navigate to the models directory in the Command Prompt and execute

    git checkout f7e99c0

5.  Run the following in the research folder
    protoc object_detection/protos/*.proto --python_out=.

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    
6.  Execute the following from within research folder. It should show 'OK'

    python object_detection/builders/model_builder_test.py
    
    
    
## Mac:
#### Reference: 
Installation instructions for protobuf v 3.4.0[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath)

1. Install Tensorflow 1.4

   pip install tensorflow==1.4

2. Install the packages

   pip install --user Cython
   pip install --user contextlib2
   pip install --user pillow
   pip install --user lxml
   pip install --user jupyter
   pip install --user matplotlib

3. Download protoc-3.4.0-osx-x86_64.zip from [here](https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)

4. Then Issue the command

   sudo unzip -o protoc-3.4.0-osx-x86_64.zip -d /usr/local bin/protoc
   
5.  Create a new directory somewhere and name it tensorflow. Clone the Tensorflow's models repo by executing

    git clone https://github.com/tensorflow/models.git

6.  Checkout the version that works with Tensorflow 1.4. Navigate to the models directory in the Command Prompt and execute

    git checkout f7e99c0
    
5.  Run the following in the research folder
    protoc object_detection/protos/*.proto --python_out=.

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim : `pwd`/object_detection
    
6.  Execute the following from withing research folder. It should show 'OK'

    python object_detection/builders/model_builder_test.py
    
    
    
# Training

##  Clone this repo to your workspace
    
    https://github.com/tamoghna21/CarND-Traffic-Light-Classifier
    
    Create 'data' and 'models' folder in the project(if they are not already present)
    
## Training is carried out in Linux
1.  sudo apt-get update
    pip install --upgrade dask
    pip install tensorflow-gpu==1.4
    
2. Follow the [setup instructions for linux](https://github.com/tamoghna21/CarND-Traffic-Light-Classifier##Linux)

## Download Data

#### Navigate inside 'data' folder

Training Dataset:

    wget https://www.dropbox.com/s/bvq7q1zoex3b46i/dataset-sdcnd-capstone.zip?dl=0
    
    unzip dataset-sdcnd-capstone.zip?dl=0
    
Validation Dataset:

    wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
    
    unzip alex-lechner-udacity-traffic-light-dataset.zip?dl=0
    
## Get the Model

#### Navigate inside 'models' folder

For Faster-RCNN Inception v2 model:

     wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
     
     tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    
For SSD Inception V2 model:

     wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz 

     tar -xvzf ssd_inception_v2_coco_2018_01_28.tar.gz.

    
## Train the model

From inside the project folder:

#### For Sim environment using Faster RCNN model:

python train.py --logtostderr --train_dir=./models/rcnn_sim
--pipeline_config_path=./config/faster_rcnn_inception_v2_coco_sim.config

#### For Udacity real environment using Faster RCNN model:

python train.py --logtostderr --train_dir=./models/rcnn_udacity --pipeline_config_path=./config/faster_rcnn_inception_v2_coco_real.config


#### For Sim environment using SSD Inception v2 model:

python train.py --logtostderr --train_dir=./models/ssd_sim
--pipeline_config_path=./config/ssd_inception_v2_coco_sim.config

#### For Udacity real environment using SSD Inception v2 model:

python train.py --logtostderr --train_dir=./models/ssd_udacity --pipeline_config_path=./config/ssd_inception_v2_coco_udacity.config


## Freeze the Graph

#### For Sim environment(Faster RCNN model):

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/faster_rcnn_inception_v2_coco_sim.config --trained_checkpoint_prefix ./models/rcnn_sim/model.ckpt-10000 --output_directory models/rcnn_sim

#### For Udacity real track environment(Faster RCNN model):

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/faster_rcnn_inception_v2_coco_real.config --trained_checkpoint_prefix ./models/rcnn_udacity/model.ckpt-10000 --output_directory models/rcnn_udacity


#### For Sim environment(SSD Inception v2 model):

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_inception_v2_coco_sim.config --trained_checkpoint_prefix ./models/ssd_sim/model.ckpt-10000 --output_directory models/ssd_sim

#### For Udacity real track environment(SSD Inception v2 model):

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_inception_v2_coco_udacity.config --trained_checkpoint_prefix ./models/ssd_udacity_udacity/model.ckpt-10000 --output_directory models/ssd_udacity






## Reference
SSD Inception [SSD Inception V2 model](https://arxiv.org/pdf/1512.00567v3.pdf).
Traffic Light Classification notebook [Traffic Light Classification notebook by alex-lechner](https://github.com/alex-lechner/Traffic-Light-Classification).

