import argparse as ap

from .core import run_hto_counting
from .utils import check_input_lengths
from .io import parse_input_csv


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


def main(args):
    args = argument_parser().parse_args(args)
    bam_files, hto_files, barcode_files, output_directories = parse_input_csv(args.csv)
    check_input_lengths(
        bam_files,
        hto_files,
        barcode_files,
        output_directories
    )
    run_hto_counting(
        bam_files,
        hto_files,
        barcode_files,
        output_directories,
        args.processes
    )
