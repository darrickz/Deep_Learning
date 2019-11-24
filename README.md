# Deep Learning

---
<font size="3">
    
Build convolutional networks for image recognition, recurrent networks for sequence generation, generative adversarial networks for image generation, and learn how to deploy models accessible from a website.
</font>

---

## Projects

### [Project 1: Dog Breed Classifier](https://github.com/darrickz/Deep_Learning/tree/master/DLND-Dog-breed-Classifier)

Built Convolutional Neural Networks (CNN) to classify 133 dog breeds using transfer learning
<table><tr>
<td>

<figure>
    <img  src="./images/dog_breed.jpg" alt="Drawing" style="width: 450px;"/>
<p align="center">      
    Predicted Dog Breed
</p>
</figure></td>

<td><figure>    
    <img  src="./images/dog_breed1.jpg" alt="Drawing" style="width: 450px;"/>
    <p align="center">Predicted to the Closest Dog Breed</p>
</figure>
  </td>  </tr></table>

---

### [Project 2: Face Generation](https://github.com/darrickz/Deep_Learning/tree/master/DLND-project-face-generation)

Trained a DCGAN on a dataset of faces.Then new images of faces that look as realistic as possble are generated:

<figure>
    <kbd>
    <img  src="./images/face_generation.png" alt="Drawing" style="height: 500 width: 1000px;"/>
    </kbd>
   <p align="center">Generated Face</p>
</figure>

---
### [Project 3: Generate TV Script](https://github.com/darrickz/Deep_Learning/tree/master/DLND-project-tv-script-generation)


RNN network is trained to generate Seinfeld TV scripts. The training data are part of the Seinfeld dataset of scripts from 9 seasons. 
<figure>
    <p align="center">    
    <kbd>
    <img  src="./images/generated_tv_script.png" alt="Drawing" style="height: 600 width: 1000px;"/>
    </kbd>
    </p>    
<p align="center">    
    Generated fake TV script
</p>
</figure>

---
### [Project 4: CNN Model Deployment](https://github.com/darrickz/Deep_Learning/tree/master/DLND-sagemaker-deployment)

This project is to build a simple web page in which a user can type in a movie review and the trained RNN model behind the scene predicts whether the review is positive or negative. The model is trained on IMDB dataset and deployed using AWS SageMaker

<figure>
    <img  src="./images/sagemaker-architecture.png" alt="Drawing" style="width: 450px;"/>
    <p align="center">  Sagemaker Architecture</p>
</figure>


Example results:
<table><tr>
<td>

<figure>
    <img  src="./images/review1.JPG" alt="Drawing" style="width: 450px;"/>    
    <p align="center">Review Predicted as Positive</p>
</figure></td>

<td><figure>    
    <img  src="./images/review2.JPG" alt="Drawing" style="width: 450px;"/>
    <p align="center">Review Predicted as Negative</p>
</figure>
  </td>  </tr></table>
