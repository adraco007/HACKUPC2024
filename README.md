# HACKUPC2024

### Procés:
- Scrapping: image_scrapping_opt
- Inicialitzar model: model_clip
- Preprocessat de dades
model clip opt no va
model no va

## Inspiration
As students of the artificial intelligence degree (from UPC), we have obviously been influenced by all the knowledge gained in the last months, where we have started to deal with neural networks, among other things. That's why we started with this challenge, as it served us to practice what we had learnt in class.

## What it does
We have implemented a web, which has two uses: to find the most similar images given a set of them. And, on the other hand, given an image of a piece of clothing (by photo or generated with AI), to obtain the most similar existing images (in the shop). All this works with pre-trained models of convolutional neural networks.

## How we built it
We have used convolutional neural networks to obtain similarities between images. Moreover, we treated the links of the images provided in the csv to classify the images into Season, ProductType and Section classes. In this way, we reduce the search space for similar images, making it more optimal.
For the web (backend), we worked with Flask. The aim of the web was to be able to visualise the results of the model more easily, as well as to see possible real applications. To achieve this, we added some extra aspects to the work. For example, we can enter images as a file (after having taken a photo, for example), but we also have a generative AI of open source clothes; and from these we can search for similar ones. In addition, we have scraped the Zara website to get the links to all the current Zara products. We do this so that, when we get a similar photo, by clicking on it we can go directly to the product on the official website (we can do this because the links of the images and the products share a certain numerical code).

## Challenges we ran into
- Coordinating frontend and backend
- Creating a good model to obtain the embeddings and finding similarities between pieces of cloth
- Dealing with the high computational and spatial cost


## Accomplishments that we're proud of
- Creating the model
- Getting a good interface
- Being able to find real products from an image of a similar product 

## What we learned
- Use of convolutional neural networks
- Dealing with embeddings


## What's next for PifIA's DLC
- Applying the knowledge obtained to ohter academic uses
