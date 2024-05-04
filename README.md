# HACKUPC2024

data_base: https://drive.google.com/file/d/1OaIzEt20LQk1ixO5UFx7wvOTZef3PKww/view?pli=1

liveshare: https://prod.liveshare.vsengsaas.visualstudio.com/join?59D31687B3449869D06577115946691FE2DF

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
Autoencoder Explicado



  Entrada de la imagen:
      La entrada de la imagen tiene una forma definida por las dimensiones de la imagen de entrada. En este caso, se asume que las imágenes son de tamaño 224x224 píxeles y tienen 3 canales de color (RGB), por lo que la forma de la entrada es (224, 224, 3).

  Encoder:
      El encoder consta de varias capas convolucionales y de pooling que reducen la dimensionalidad de la imagen de entrada. En este autoencoder, se aplican capas de convolución seguidas de capas de pooling para extraer características de la imagen. La reducción de dimensionalidad se realiza mediante el uso de capas de pooling, que reducen el tamaño espacial de la imagen y el número de canales de características.
      La forma de salida del encoder, antes del cuello de botella, es determinada por la última capa de pooling, que en este caso es (28, 28, 8), donde 8 es el número de canales de características.

  Cuello de botella (Bottleneck):
      El cuello de botella es una capa densa que contiene los embeddings de las características más importantes de la imagen. En este autoencoder, se usa una capa densa con 128 neuronas para capturar estas características. La forma de salida del cuello de botella es (128,), ya que es un vector de características de longitud 128.

  Decoder:
      El decoder es responsable de reconstruir la imagen original a partir de los embeddings en el cuello de botella. En este autoencoder, el decoder consta de capas densas y capas de convolución transpuestas (up-sampling) que aumentan la dimensionalidad de las características.
      La forma de salida del decoder es la misma que la forma de entrada de la imagen original, es decir, (224, 224, 3), ya que el objetivo es reconstruir una imagen RGB de tamaño 224x224 píxeles.





--------------------------------------------------------------------------------------------------------------------
File Hierarchy:

- ./data: contains the data used for both preprocessing, training and validating the different models
- ./models: contains different binaries with the trained models
- ./src: scripts