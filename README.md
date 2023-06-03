# AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model
Code for our ACL-2023 paper [AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model](https://arxiv.org/abs/2305.16734)

## Environment
- Python==3.7
- PyTorch==1.11.0
- ipdb==0.13.9
- transformers==4.18.0 (pip)
- tensorboardx==2.5.1 (pip)
- sentencepiece==0.1.96 (pip)
- penman==1.2.2 (pip)
- networkx==2.6.3 (pip)
- amrlib==0.7.1 (pip)

Or use the yml file we provide.

### AMR Parser Setup
We follow the instruction in https://amrlib.readthedocs.io/en/latest/install/ to
install the trained AMR parser model. We use `parse_spring` version 0.1.0 as our
parser

## Data
We support `ace05e`, and `ere`.

### Preprocessing
Our preprocessing mainly adapts [OneIE's](https://blender.cs.illinois.edu/software/oneie/) and [DEGREE's](https://github.com/PlusLabNLP/DEGREE) released scripts with minor modifications. We deeply thank the contribution from the authors of the paper.

#### `ace05e`
1. Prepare data processed from [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event)
2. Put the processed data into the folder `processed_data/ace05e_dygieppformat`
3. Run `./scripts/process_ace05e.sh`

#### `ere`
1. Download ERE English data from LDC, specifically, "LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2", "LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2", "LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2"
2. Collect all these data under a directory with such setup:
```
ERE
├── LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
│     ├── data
│     ├── docs
│     └── ...
├── LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
│     ├── data
│     ├── docs
│     └── ...
└── LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
      ├── data
      ├── docs
      └── ...
```
3. Run `./scripts/process_ere.sh`

The above scripts will generate processed data in `./process_data`.

## Training
Run `./scripts/train_eae.sh`

### Clean up log when training
If you want to have more clean log file, you can comment out line 151 in the "layout.py" file in penman package
```
logger.info('Interpreted: %s', g)
```
This should be in `CONDAENVPATH/envs/Ampere/lib/python3.7/site-packages/penman`


## Citation

If you find that the code is useful in your research, please consider citing our paper.

    @inproceedings{acl2023ampere,
        author    = {I-Hung Hsu and Zhiyu Xie and Kuan-Hao Huang and Premkumar Natarajan and Nanyun Peng},
        title     = {AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model},
        booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
        year      = {2023},
    }

## Contact

If you have any issue, please contact I-Hung Hsu at (ihunghsu@usc.edu)
