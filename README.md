# SemanticEncoder

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
├── featureextraction/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataparsing.py
│   │   └── imgftsextraction.py
│   ├── test/
│   │   ├── mockfiles/
│   │   │   ├── AWA2/
│   │   │   └── CUB200/
│   │   ├── __init__.py
│   │   ├── test_dataparsing.py
│   │   └── test_imgftsextraction.py
│   ├── __init__.py
│   └── main.py
├── README.md
├── requirements_local.txt
└── requirements_server.txt
```

### Feature Extraction

The application feature extraction parses the semantic features of AwA2 and CUB 200 data sets and runs ResNet 50 to 
extract each set visual features.   
      