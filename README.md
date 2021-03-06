# SmartSight

Distributing deep neural networks (DNN) to Edge-to-MCU and applying quantization at layer granularity. We use different TensorFlow Lite supported MCU platforms, representative of to-day’s state-of-the-art MCUs. We use NVIDIA Jetson AGX Xavier as our edge platform. 

## List of MCU
  1. [Wio Terminal: ATSAMD51](https://www.seeedstudio.com/Wio-Terminal-p-4509.html)
  2. [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview)
  3. [Himax WE-I Plus EVB Endpoint AI Development Board](https://www.sparkfun.com/products/17256)
  4. [HiLetgo ESP32](http://www.hiletgo.com/ProductDetail/2157143.html)

## Edge Device
  * [NVIDIA Jetson AGX Xavier Development Kit](https://www.seeedstudio.com/NVIDIA-Jetson-AGX-Xavier-Development-Kit-p-4418.html?gclid=CjwKCAjw_JuGBhBkEiwA1xmbRZFk_-s8W7FWp_Q8OKIi7QEnQR4cOn2ftUXtMS-khYo-XDTThSezBxoCGLMQAvD_BwE)

## Current implementation
  > Wio Terminal [Get Started](https://wiki.seeedstudio.com/Wio-Terminal-Getting-Started/)\
  > TF_Lite to .h coversion on Ubuntu `xxd -i Filename.ttf > Filename.h`

## Requirements
  > `pip install tf-nightly` \
  > `pip install keras` \
  > `pip install --user --upgrade tensorflow-model-optimization`

## Arduino libraries 
  > [EloquentTinyML](https://www.google.com) version 0.0.7\
  > [Arduino_TensorFlowLite](https://www.tensorflow.org/lite/microcontrollers) version 1.15.0 ALPHA

