# VSE - Visual Semantic Encoder

This project aims to verify the feasibility of using an autoencoder to learn a new distribution for a set of image's 
visual features, associated with semantic features, that better represent each image and improve discriminalty among 
classes. 

The visual features are extracted using a convolutional neural network such as ResNet50 or GoogLeNet. The semantic 
features are acquired from attributes indicated by specialists that describe each class.

Our approach proposes to merge the semantic and visual features in a single encoding with a condensed dimensionality. 
This project contains several tests that were made through our research, including simple image classification and 
Zero Shot Learning classification. 

The research is made by Damares Resende, a Computer Science Master Student from the University of São Paulo, and Moacir 
A. Ponti, her teacher advisor.

## Project Structure

```buildoutcfg
/SemanticEncoder/
├── encoders/
│   ├── sae/
│   │   ├── src
│   │   │   ├── __init__.py
│   │   │   ├── awa_demo.py
│   │   │   └── cub_demo.py
│   │   ├── test
│   │   │   ├── __init__.py
│   │   │   ├── test_awa_demo.py
│   │   │   └── test_cub_demo.py
│   │   └── __init__.py
│   ├── vse/
│   │   ├── src
│   │   │   ├── __init__.py
│   │   │   └── autoencoder.py
│   │   ├── test
│   │   │   ├── __init__.py
│   │   │   └── test_autoencoder.py
│   │   └── __init__.py
│   ├── tools/
│   │   ├── src
│   │   │   ├── __init__.py
│   │   │   ├── sem_analysis.py
│   │   │   └── utils.py
│   │   ├── test
│   │   │   ├── mockfiles
│   │   │   ├── __init__.py
│   │   │   ├── test_sem_analysis.py
│   │   │   └── test_utils.py
│   │   └── __init__.py
│   └── __init__.py
├── featureextraction/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataparsing.py
│   │   ├── featureextraction.py
│   │   └── matlabparser.py
│   ├── test/
│   │   ├── mockfiles/
│   │   │   ├── AWA2/
│   │   │   └── CUB200/
│   │   ├── __init__.py
│   │   ├── test_dataparsing.py
│   │   ├── test_featureextraction.py
│   │   └── test_matlabparser.py
│   ├── __init__.py
│   └── extractfeatures.py
├── README.md
├── requirements_local.txt
└── requirements_server.txt
```

### Feature Extraction

The application feature extraction parses the semantic features of AwA2 and CUB 200 data sets and runs ResNet 50 to 
extract each set visual features.   
     
### Encoders

The application encodes semantic features and analyzes space formed for Zero Shot Learning classification and 
SVM classification. SAE is the baseline project, published by Elyor Kodirov, Tao Xiang, and Shaogang Gong in the 
paper "Semantic Autoencoder for Zero-shot Learning". SEEC is the proposed solution. 
