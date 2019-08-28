# ProD

Article:

>[Fancy article name]()

>Maarten Van Brempt, Jim Clauwaert, Willem Waegeman, Marjan De Mey

>Important article

## Introduction
The Promoter Designer (ProD) tool is a shallow neural network created by Van Brempt M. and Clauwaert. J et. al. for the prediction of promoter strength or transcription initiation frequency by variation of the spacer sequence. 
It is created to construct promoter libraries, further exploiting biological capabilities of the microorganisms that allow for the fine-tuning of genetic circuits. A neural network has been trained on hundreds of thousands of sequences that have been randomized in the **17nt spacer sequence**. Therefore, generated promoters, ranging from no expression (strengh: `0`) to high expression (strength: `10`) all feature the same UP-region, binding boxes (-35, -10) and untranslated region (UTR).

**PROMOTER MAP**

`
[UP-region][-35-box][spacer][-10-box][ATATTC][UTR]
`

`
[GGTCTATGAGTGGTTGCTGGATAAC][TTTACG][NNNNNNNNNNNNNNNNN][TATAAT][ATATTC][AGGGAGAGCACAACGGTTTCCCTCTACAAATAATTTTGTTTAACTTT]
`


## Usage
The tool can be used through Binder or on a local system.
1. **Binder**
Binder offers the hosting of GitHub on their servers

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdcla/ProD/master?filepath=ProD_Notebook.ipynb)

2. **Local installation**

To run ProD on a local machine, simply clone the repository in your working directory and install the necessary python libraries:

	git clone https://github.com/jdcla/FlowCyt.git
	conda env create -f environment.yml

No CUDA support is necessary (GPU accelarated processing), a PyTorch installation supporting CUDA is therefore not installed through `environment.yml`. See [www.pytorch.com](www.pytorch.com) for more information.

Local installations can run the python script `ProD.py`

```
usage: ProD.py [-h] [--output_path OUTPUT_PATH] [--lib] [--lib_size LIB_SIZE]
               [--classes [CLASSES [CLASSES ...]]] [--cuda]
               input_data

Promoter Designer (ProD) tool

positional arguments:
  input_data             location of the text file containing spacer sequences

optional arguments:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        location of the text file containing spacer sequences
  --lib                 create library from blueprint
  --lib_size LIB_SIZE, -ls LIB_SIZE
                        size of each class in library (min:1, max:64)
  --strengths [STRENGTHS [STRENGTHS ...]],
                        promoter strengths included in the library (min:0, max:10)
  --cuda                use CUDA. Requires PyTorch installation supporting
                        CUDA!

```



## Creating a Library

To create a custom promoter library, a single input blueprint is given that functions as the source from which spacer sequences are evaluated. The tool will run through the following steps:

1. Create all possible sequences from the degenerate input sequence
2. Determine the promoter strengths, retain all spacer sequences for requested promoter strengths
3. Sample promoters to construct library.
4. Construct degenerate sequence (library blueprint) from all sequences. For each blueprint, the fraction of sequences classified to each category of strength is given.

If the amount of sequences possible from the input sequence exceeds 500,000, spacers will be sampled (100,000) instead and no library blueprint is created. To attain feasible processing times, a minimal amount of user guidance in the construction of the library blueprint is required.

**NOTE:** Promoter strength is divided in 11 ordinal classes ranging from 0 to 10. Overlap between neighbouring class strengths is expected. Therefore, when constructing a library it can be beneficial to group classes together. Specifically, we recommend the following interpretation of four sets of input strengths.
* zero to low expression: `strengths = 0 1 2`
* low to medium expression: `strengths = 3 4 5`
* medium to high expression: `strengths =  6 7 8`
* high to very high expression: `strengths = 9 10`

**EXAMPLE**

`
python ProD.py NNNCGGGNCCNGGGNNN --lib --strengths 8 9 10
`

## Evaluate Custom Spacers
It is possible to evaluate custom sequences. The input can be given as a list or the path to a fasta file.

**EXAMPLE**

`
python ProD.py NNNCGGGNCCNGGGAAA AAACCCGGGTTTTAAAC
`

`
python ProD.py ex_seqs.fa
`

