import logging
import scipy


def subset_called_cells(
        counts: scipy.sparse.csr_matrix,
        cell_barcodes: list[str],
        called_barcodes: set[str]
) -> tuple[scipy.sparse.csr_matrix, list[str]]:
    """
    subsets count matrix to contain only cell barcodes retained during cellranger filter step

    :param counts:              sparce count matrix of shape len(cell_barcodes) x HTOs
    :param cell_barcodes:       list of barcodes corresponding to count matrix rows
    :param called_barcodes:     set of filtered cell barcodes

    :return:                    subset of counts of shape len(called_barcodes) x HTOs, list of retained cell barcodes
    """
    logging.info('subsetting counts')

    called: list[bool] = []
    retained_barcodes: list[str] = []
    for cell_barcode in cell_barcodes:
        barcode_called = cell_barcode in called_barcodes
        called.append(barcode_called)
        if barcode_called:
            retained_barcodes.append(cell_barcode)

    return counts[called, :], retained_barcodes


def count_dict_to_csr_matrix(
        count_dict: dict[str, dict[str, set[str]]],
        htos: list[tuple[str]]
) -> tuple[scipy.sparse.csr_matrix, list[str], list[str]]:
    """
    convert count_dict to csr_matrix

    :param count_dict:  dictionary of dictionary of sets containing umi counts per HTO per cell barcode
    :param htos:        list of key, value tuples where key is the HTO sequence and value is the HTO name

    :return:            sparce count matrix of shape cell_barcodes x HTOs, list of cell barcodes, list of HTO names
    """
    cell_barcodes: list[str] = []
    hto_counts_per_cell_barcode: list[list[int]] = []
    for cell_barcode, umis_per_hto in count_dict.items():
        hto_counts_per_cell_barcode.append(
            [len(umis_per_hto[sequence]) if sequence in umis_per_hto else 0 for sequence, name in htos]
        )
        cell_barcodes.append(cell_barcode)

    count_matrix = scipy.sparse.csr_matrix(hto_counts_per_cell_barcode)
    count_matrix.eliminate_zeros()

    return count_matrix, cell_barcodes, [hto_name for _, hto_name in htos]


def check_input_lengths(*args):
    input_length = len(args[0])
    for arg in args:
        if not input_length == len(arg):
            raise ValueError('input file lists must have the same length')

        input_length = len(arg)
