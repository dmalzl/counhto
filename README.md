# counhto

![img](https://img.shields.io/badge/pypi-1.1.0-blue)

An easy to use tool to count hashtag oligos (HTOs) from 10x cellranger count output processed with Antibody Captures for sample multiplexing and assign tags to cells using the cellranger Jibes algorithm (this part is a minimal edit of the original algorithm as implemented in cellranger > 6.0)

## Install
The most convenient and easy way to install the package is
```
pip install counhto
```

alternatively you could also clone the repository and install it manually like
```
git clone git@github.com:dmalzl/counhto.git
cd countho
pip install .
```

## Usage
Using it is as simple as setting up a csv file with the following structure
|bamfile|barcodefile|htofile|outputdir|
|:------|:----------|:------|:--------|
|cellranger/outs/possorted_bam.bam|cellranger/outs/filtered_feature_bc_matrix/barcodes.tsv.gz|cellranger/outs/feature_ref.csv|/path/to/outputdir/|
|cellranger/outs/possorted_bam.bam|cellranger/outs/filtered_feature_bc_matrix/barcodes.tsv.gz|cellranger/outs/feature_ref.csv|/path/to/outputdir/|

and invoking countho as follows
```
countho --csv sample_csv.csv [-p n]
```

The `-p` argument specifies the number of cpus to use for processing however this only has an effect if more than one samples are supplied.

counhto then counts UMIs per HTO and automatically performes tag assignment using cellrangers Jibes algoritm (see [cell multiplexing documentation](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/algorithms/cellplex) for more info). The output for each sample will then be written to the specified output directories where the barcodes.tsv file contains the tag assignment information. The directory has following structure
```
/path/to/outputdir/
|__
   |__barcodes.tsv  # filtered barcodes with tag assignment information
   |__features.tsv  # names of the HTOs as specified in the feature_ref.csv file
   |__matrix.mtx    # MatrixMarket formated count matrix of shape n_barcodes x n_HTOs
```
