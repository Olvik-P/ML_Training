import tensorflow as tf
# import torch

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Torch сказал - GPU name:", torch.cuda.get_device_name(0))
#     print("Torch сказал - Number of GPUs:", torch.cuda.device_count())

print("Tensorflow сказал - GPU devices:",
      tf.config.list_physical_devices('GPU'))
