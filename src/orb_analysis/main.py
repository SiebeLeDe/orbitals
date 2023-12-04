import argparse
import pathlib as pl
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser  # replace "some_module" with the actual module name


def main():
    parser = argparse.ArgumentParser(description='Parser for the adf.rkf file to analyze.')
    parser.add_argument('--file', type=str, help='The calculation file (adf.rkf) to analyze')
    parser.add_argument('--spin', type=str, help='The spin to analyze. Options are "A" and "B"', required=False)
    parser.add_argument('--orb_range', type=int, nargs=2, help='The range of orbitals to analyze from HOMO-x - LUMO+x, e.g. --orb_range 5, 5', required=False)
    parser.add_argument('--irrep', type=str, help='The irrep to analyze', required=False)
    parser.add_argument('--output_file', type=str, help='Path to the output file', required=False)

    args = parser.parse_args()

    orb_range = args.orb_range if args.orb_range is not None else (6, 6)
    analyzer = create_calc_analyser(args.file)
    analysis = analyzer(orb_range=orb_range, spin=args.spin, irrep=args.irrep)

    if args.output_file:
        pl.Path(args.output_file).write_text(analysis)
    else:
        print(analysis)


if __name__ == "__main__":
    main()
