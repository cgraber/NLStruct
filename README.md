# NLStruct
Code used to produce experimental results for the paper ["Deep Structured Prediction with Nonlinear Output Activations"](https://arxiv.org/abs/1811.00539)

Welcome! 

## Requirements:
This code was developed using the following versions of the following libraries. I cannot guarantee that it will work with other versions:
- Python 3.5.2
- numpy 1.15.2
- PyTorch 0.4.0
- torchvision 0.2.0
- Matplotlib 2.2.2 (though you can just comment these parts out if necessary)
- Pillow 5.1.0 (for experiments requiring images)
- scikit-image
- dill

Additionally, running this code requires compiling a python extension written in C++ - this will require Python dev libraries as well as a C++ compiler. To compile this extension, enter the 'deepstruct/fastmp' directory and execute:
`python setup.py build`

## Structure
The python scripts used to run the experiments are found in the experiments folder, while shared model code is in the deepstruct folder.

## Data
The data used for the word recognition experiments can be found in the data folder. The other datasets can be downloaded from various locations:
- [Weizmann Horse Database](http://www.msri.org/people/members/eranb/)
- [MIRFLICKR25k](http://press.liacs.nl/mirflickr/mirdownload.html)
- [Bibtex/Bookmarks](http://mulan.sourceforge.net/datasets-mlc.html)


## Short disclaimer
This repo consists of code that was taken from a larger research codebase. I attempted to prune everything so that only the relevant code remains; if something seems to be missing, just let me know and I'll make sure to track it down and add it back in. Thanks for taking a look!
