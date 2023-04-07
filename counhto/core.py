import logging
import scipy

import pysam as ps

from .utils import (
    count_dict_to_csr_matrix,
    subset_called_cells
)
from .io import (
    read_hto_file,
    read_called_barcodes
)

from collections import defaultdict


def count_capture_per_cell_barcode(
        path_to_bamfile: str,
        htos: list[tuple[str]]
) -> tuple[scipy.sparse.csr_matrix, list[str], list[str]]:
    """
    parses a bamfile generated by cellranger and counts UMIs per HTO for each cell barcode
    assumes cepture library reads are unaligned reads in the given bamfile

    :param path_to_bamfile:     path to possorted_bam.bam as generated by cellranger
    :param htos:                list of key, value tuples where key is the HTO sequence and value is the HTO name

    :return:                    sparse count matrix,
                                cell barcodes corresponding to rows in counts,
                                HTO names corresponding to columns in counts
    """
    log_msg = f'processing bam_file for HTO counts: {path_to_bamfile}'
    logging.info(f'start {log_msg}')

    bam = ps.AlignmentFile(path_to_bamfile, 'rb')
    feature_umis_per_cell_barcode: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for capture_read in bam.fetch('*'):
        barcodes = {}
        for tag, alternative_tag, tag_label in zip(
                ['CB', 'fb', 'UB'],
                ['CR', 'fr', 'UR'],
                ['cell', 'feature', 'umi']
        ):
            corrected = capture_read.has_tag(tag)
            tag_to_fetch = tag if corrected else alternative_tag

            if not capture_read.has_tag(tag_to_fetch):
                continue

            barcode = capture_read.get_tag(tag_to_fetch)

            if tag_label == 'CB' and not corrected:
                barcode = barcode + '-1'

            barcodes[tag_label] = barcode

        if 'feature' not in barcodes:
            continue

        feature_umis_per_cell_barcode[barcodes['cell']][barcodes['feature']].add(barcodes['umi'])

    bam.close()

    count_matrix, cell_barcodes, feature_names = count_dict_to_csr_matrix(
        feature_umis_per_cell_barcode,
        htos
    )

    logging.info(f'done {log_msg}')

    return count_matrix, cell_barcodes, feature_names


def make_hto_count_matrix(
        path_to_bam_file: str,
        path_to_hto_file: str,
        path_to_called_barcodes_file: str,
) -> tuple[scipy.sparse.csr_matrix, list[str], list[str]]:
    """
    takes the path to the possorted_bam.bam file, the used feature_ref.csv file and the generated filtered barcodes.tsv
    file and generates a tag counts matrix of shape n_filtered_barcodes x n_features

    :param path_to_bam_file:                path to possorted_bam.bam as generated by cellranger
    :param path_to_hto_file:                path to the used feature_ref.csv file used to generate possorted_bam.bam
    :param path_to_called_barcodes_file:    path to the filtered barcodes.tsv file
    :return:                                tag count matrix, barcodes corresponding to rows and
                                            features correspinding to columns
    """
    htos = read_hto_file(path_to_hto_file)
    counts, barcodes, features = count_capture_per_cell_barcode(
        path_to_bam_file,
        htos
    )

    compressed = True if path_to_called_barcodes_file.endswith('gz') else False
    called_barcodes = read_called_barcodes(
        path_to_called_barcodes_file,
        compressed=compressed
    )
    subset_counts, subset_barcodes = subset_called_cells(
        counts,
        barcodes,
        called_barcodes
    )
    return subset_counts, subset_barcodes, features
