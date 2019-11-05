# M2MNet
**M2MNet:Multiframe to Multiframe Network for Video Denoising with Spatial and Temporal Convolution**

***code will be upload soon.***

# Requirements:


# Code usage
The method reads the video as a sequence of images. The sequence of images is generated using ffmpeg.

# Train
Using generate_patches.py to process generate training file (.npy file).
**Training:** python main.py

**Note:** you can add command line arguments according to the source code, for example：
python main.py --batch_size 32
python main.py --lr 0.0001

# Test
***Example:*** denoising color video with noise level (sigma) of 20.
python main.py --phase test –is_color 1 –sigma 20 –paras checkpoing/color_20 –test_path data/tennis

***Example:*** denoising grayscale video with noise level (sigma) of 10.
python main.py --phase test –is_color 0 –sigma 10 –paras checkpoing/grayscale_10 –test_path data/tennis
