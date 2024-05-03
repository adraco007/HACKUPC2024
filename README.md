# HACKUPC2024

data_base: https://drive.google.com/file/d/1OaIzEt20LQk1ixO5UFx7wvOTZef3PKww/view?pli=1

liveshare: https://prod.liveshare.vsengsaas.visualstudio.com/join?86B0D29196D3A775E8B8D3ADC9A1710530D2

Image Computing for an Ecommerce
Given a dataset of garment images from various angles, the challenge is to develop an algorithm that identifies 
duplicated or very similar images not belonging to the same set (each set consists of three consecutive photos). 
This involves comparing colors, features, and bitmaps.

The task is computationally intensive, requiring over 8 billion computations due to the combinatorial complexity of 
image comparisons in three dimensions.

The algorithm's accuracy and speed will be key evaluation criteria. Advanced teams may use provided photos to create
 a website showcasing their work. Senior teams could infer garment details (year, season, indicators) from generated
  URLs.


--------------------------------------------------------------------------------------------------------------------
File Hierarchy:

- ./data: contains the data used for both preprocessing, training and validating the different models
- ./models: contains different binaries with the trained models
- ./src: scripts