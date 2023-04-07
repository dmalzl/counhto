import sys
import logging

import argparse as ap
import multiprocessing as mp

from .count import make_hto_count_matrix
from .utils import check_input_lengths, fill_unassigned_cells_entries
from .io import parse_input_csv, write_outputs
from .marginal import get_marginal_tag_assignment
from .jibes import get_jibes_tag_assignment

FORMAT = '%(asctime)s %(message)s'


def argument_parser():
    parser = ap.ArgumentParser(
        formatter_class=ap.RawDescriptionHelpFormatter,
        description="""
            This tool is dedicated to counting hashtag oligos (HTOs) per cell barcode from 10x cellranger count output 
            with antibody capture libraries which is necessary in some cases because cellranger multi does not handle
            5'-chemistry libraries. In brief, this tool expects the possorted bam file and filtered barcodes.tsv file
            as generated by cellranger as well as the used feature reference csv and a output destination folder and
            will attempt to generate a filtered cell barcode x n HTOs count matrix which is written to the output
            given output folder. Each argument can handle multiple files which are matched by position so make sure that
            your supplied files line up when doing so.
        """,
        epilog='example usage:\n countho --csv samples.csv',
        conflict_handler='resolve'
    )
    required = parser.add_argument_group('Required arguments')
    required.add_argument(
        '--csv',
        metavar='SAMPLECSV',
        required=True
    )

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument(
        '--processes', '-p',
        help='number of cores to use to process the input data. only has an effect if more than one samples are given',
        default=1,
        type=int
    )

    return parser


def process_sample(path_dict: dict[str, str]):
    path_to_bam_file = path_dict['bam_file']
    path_to_hto_file = path_dict['hto_file']
    path_to_called_barcodes_file = path_dict['barcode_file']
    output_path_prefix = path_dict['output_directory']

    tag_counts_matrix, barcodes, feature_names = make_hto_count_matrix(
        path_to_bam_file,
        path_to_hto_file,
        path_to_called_barcodes_file
    )

    marginal_features_per_cell_table = get_marginal_tag_assignment(
        tag_counts_matrix,
        feature_names,
        barcodes
    )

    jibes_features_per_cell_table = get_jibes_tag_assignment(
        tag_counts_matrix,
        feature_names,
        barcodes,
        marginal_features_per_cell_table
    )

    jibes_features_per_cell_table = fill_unassigned_cells_entries(
        jibes_features_per_cell_table,
        barcodes,
        tag_counts_matrix
    )

    write_outputs(
        tag_counts_matrix,
        jibes_features_per_cell_table,
        feature_names,
        output_path_prefix
    )


def main():
    args = None
    if len(sys.argv) == 1:
        args = ["--help"]

    args = argument_parser().parse_args(args)

    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    bam_files, hto_files, barcode_files, output_directories = parse_input_csv(args.csv)
    check_input_lengths(
        bam_files,
        hto_files,
        barcode_files,
        output_directories
    )
    iterable = [
        {'bam_file': bam_file, 'hto_file': hto_file, 'barcode_file': barcode_file, 'output_directory': output_directory}
        for bam_file, hto_file, barcode_file, output_directory in
        zip(bam_files, hto_files, barcode_files, output_directories)
    ]
    if args.processes > 1:
        p = mp.Pool(args.processes)
        _map = p.map

    else:
        _map = lambda func, iterable: list(map(func, iterable)) # necessary to invoke map

    _map(process_sample, iterable)

    if args.processes > 1:
        p.close()
