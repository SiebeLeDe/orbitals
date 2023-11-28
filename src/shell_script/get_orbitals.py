import argparse
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser  # replace "some_module" with the actual module name


def main():
    parser = argparse.ArgumentParser(description='Process some file.')
    parser.add_argument('--file', type=str, help='The file to process')

    args = parser.parse_args()

    analyzer = create_calc_analyser(args.file)
    print(analyzer.get_sfo_orbital_energy(1, "1_A1"))


if __name__ == "__main__":
    main()
