# ProD

## Introduction
The Promoter Designer (ProD) tool is a shallow neural network created by Van Brempt M. and Clauwaert. J et. al. for the prediction of promoter strength or transcription initiation frequency by variation of the spacer sequence. 

Article:

>[Fancy article name]()

>Maarten Van Brempt, Jim Clauwaert, Willem Waegeman, Marjan De Mey

>Important article

## Usage
The tool can be used through Binder or with a local installation.
1. **Binder**
Usage through Binder does not require a local installation.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdcla/ProD/master?filepath=ProD_Notebook.ipynb)

2. **Local installation**

To run ProD on a local machine, simply clone the repository in your working directory and install the necessary python libraries:

	git clone https://github.com/jdcla/FlowCyt.git
	conda env create -f environment.yml

Local installations can run the python script `ProD.py`

```
python ProD.py -h

	usage: ProD.py [-h] [--output_path OUTPUT_PATH] [--cuda] data_path

	Promoter Designer tool

	positional arguments:
	  data_path             location of the text file containing spacer sequences

	optional arguments:
	  -h, --help            show this help message and exit
	  --output_path OUTPUT_PATH, -o OUTPUT_PATH     
                            location of the text file containing spacer sequences
	  --cuda                use CUDA. Requires PyTorch installation supporting CUDA!
```

**EXAMPLE**

`
python ProD.py ex_seqs.fa -o ex_seqs.out
`

No CUDA support is necessary (GPU accelarated processing), a PyTorch installation supporting CUDA is therefore not installed through `environment.yml`. See [www.pytorch.com](www.pytorch.com) for more information.





