# Serinity :brain:
A simple introduction to EEG using mind letter recognition.

Hardware used : [Epoc Flex](https://www.emotiv.com/product/epoc-flex-gel-sensor-kit/)

# Table of Content
Introduction\
Quickstart\
What it does concretely\
How it works\
Datasets\
Documentations\
Used libs

# Introduction
Serinity is a simple mind letter recognition system made to discover EEG by 3 students, or in other term, it transposes the letter we think about (Visually as an image) into an extern program.\
This program can mainly serve as a documentation and a good start to EEG and brainwaves analysis now and less as a pratical use.\
We'll trace our path to get everything we came accross for a better understanding and I hope a better approch to make this abordable.

⚠️ The used trained model (AI) might not work correctly on other people due to our lack of data. This can surely be upgraded using crossfit training and wider data.


##### Table of Contents  
[Introduction](#Introduction)  
[Emphasis](#emphasis)  
...snip...    
<a name="Introduction"/>
## Headers

## What does our program do concretely ?
Our program takes data (after its training), and deduct the letter on the screen, and in the better state create clear words.\
This will be more clearly decided later on.

<a href="https://ibb.co/MCSK9J7"><img src="https://i.ibb.co/3CdVkPc/Sans-titre.png" alt="Sans-titre" border="0" /></a>

## How it works
The program itself get the data from the headset using [LSL](https://labstreaminglayer.readthedocs.io/info/intro.html) or older data, preprocesses to make the data clearer and more usable and then process it using a CNN model.\
Most of this processes are automated using EEG library, which will be explained more below.\
Listes des processes

## How datasets were made
All datasets used were made by an amazing student named [Paul](https://github.com/PaulAncey), he gives his head to science and that what makes him an hero.\
They can be public with his agreement.

# Our parcour
As normal Epitech students, we discover and learn the world of brainwaves analysis and electroencephalography with this project, and that is undoubtely the fact we got some difficulties. This project was extremely interessting in its form with no doubt, but was really difficult to abord in its background. We discover for the majority how AI works with this project, and mostly tries to figures out what everything means. Here's the reason why we would like to recommend you some specific resources that can help discover the amazing world of Electroencephalography (EEG is simpler) :

### Artificial Intelligence
* [Aboarding CNN and RNN](https://stanford.edu/~shervine/l/fr/teaching/cs-230/pense-bete-reseaux-neurones-convolutionnels)

### EEG
* [The complete guide on electroencephalography](https://imotions.com/guides/electroencephalography-eeg/)
* [Understanding the concept of EEG data processing](https://www.bitbrain.com/blog/ai-eeg-data-processing)
* [Preprocessing in EEG](http://learn.neurotechedu.com/preprocessing/)
* [Public Datasets](https://github.com/meagmohit/EEG-Datasets)


## Maintainers
[David](https://github.com/Davphla)\
[Alexis](https://github.com/GaliMouette)\
[Théo](J'ai Pas trouvé)
