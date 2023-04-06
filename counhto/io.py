import gzip
import os
import logging
import scipy


def parse_input_csv(path_to_csv_file: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    parses the input csv file and returns a tuple of lists of bamfiles, htofiles, barcodefiles and outputdir

    :param path_to_csv_file:    csv file containing four columns with header bamfile,barcodefile,htofile,outputdir

    :return:                    list of bamfiles, list of htofiles, list of barcodefiles, list of outputdirs
    """

    with open(path_to_csv_file, 'r') as csv_file:
        header = csv_file.readline().rstrip().split(',')
        bam_files: list[str] = []
        barcode_files: list[str] = []
        hto_files: list[str] = []
        output_dirs: list[str] = []
        for sample_line in csv_file:
            for entry, entry_list in zip(
                sample_line.rstrip().split(','),
                [bam_files, barcode_files, hto_files, output_dirs]
            ):
                entry_list.append(entry)

    return bam_files, hto_files, barcode_files, output_dirs


def read_hto_file(path_to_hto_file: str) -> list[tuple[str]]:
    """
    read feature_ref csv file as supplied to cellranger when using antibody capture libraries

    :param path_to_hto_file:    path to feature_ref file

    :return:                    list of key, value tuples where key is the HTO sequence and value is the HTO name
    """
    logging.info(f'reading HTO sequences and names from {path_to_hto_file}')

    htos: list[tuple[str]] = []
    with open(path_to_hto_file, 'r') as hto_file:
        header = hto_file.readline().rstrip().split(',')
        name_idx = header.index('name')
        barcode_idx = header.index('sequence')
        for line in hto_file:
            fields = line.rstrip().split(',')
            htos.append(
                tuple(fields[idx] for idx in [barcode_idx, name_idx])
            )

    return htos


def read_called_barcodes(
        path_to_barcode_file: str,
        compressed: bool = True
) -> set[str]:
    """
    read called barcodes from filtered barcodes.tsv files as returned by cellranger

    :param path_to_barcode_file:    path to filtered barcodes.tsv file
    :param compressed:              indicates if barcodes file is gzipped

    :return:                        set of filtered cell barcodes present in the final data
    """
    logging.info(f'reading called barcodes for count subsetting from {path_to_barcode_file}')

    _open = gzip.open if compressed else open
    mode = 'rt' if compressed else 'r'
    with _open(path_to_barcode_file, mode) as barcode_file:
        called_barcodes = set(
            barcode.rstrip() for barcode in barcode_file
        )

    return called_barcodes


def write_counts(
        counts: scipy.sparse.csr_matrix,
        barcodes: list[str],
        features: list[str],
        output_path_prefix: str
) -> None:
    """
    write results to files

    :param counts:              sparse count matrix
    :param barcodes:            list of cell barcodes corresponding to count matrix rows
    :param features:            list of HTO names corresponding to count matrix columns
    :param output_path_prefix:  path to output directory

    :return:                    None
    """
    logging.info(f'writing output to {output_path_prefix}')

    if not os.path.exists(output_path_prefix):
        os.mkdir(output_path_prefix)

    scipy.io.mmwrite(
        os.path.join(output_path_prefix, 'matrix.mtx'),
        counts
    )

    for filename, data in zip(
            ['barcodes.tsv', 'features.tsv'],
            [barcodes, features]
    ):
        with open(os.path.join(output_path_prefix, filename), 'w') as ofile:
            for item in data:
                ofile.write(item + '\n')
