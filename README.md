# Image Captioning Project

This project implements an image captioning system using deep learning techniques. The system is capable of generating textual descriptions for images using a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs).


## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Create a conda environment:**
    ```bash
    conda create -p img_cap_env python=3.9
    ```

3. **Activate the conda environment:**
    ```bash
    conda activate ./img_cap_env
    ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

**Download the Flickr8k dataset:**
    - [Images](https://forms.illinois.edu/sec/1713398)
    - [Captions](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)