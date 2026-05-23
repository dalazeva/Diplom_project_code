import csv
import argparse


COLUMNS = [
    "menopaus",
    "agegrp",
    "density",
    "race",
    "Hispanic",
    "bmi",
    "agefirst",
    "nrelbc",
    "brstproc",
    "lastmamm",
    "surgmeno",
    "hrt",
    "invasive",
    "cancer",
    "training",
    "count",
]


MAPS = {
    "menopaus": {
        "0": "premenopausal",
        "1": "postmenopausal_or_age_55_or_more",
        "9": "unknown",
    },
    "agegrp": {
        "1": "35-39",
        "2": "40-44",
        "3": "45-49",
        "4": "50-54",
        "5": "55-59",
        "6": "60-64",
        "7": "65-69",
        "8": "70-74",
        "9": "75-79",
        "10": "80-84",
    },
    "density": {
        "1": "almost_entirely_fat",
        "2": "scattered_fibroglandular_densities",
        "3": "heterogeneously_dense",
        "4": "extremely_dense",
        "9": "unknown_or_different_measurement_system",
    },
    "race": {
        "1": "white",
        "2": "asian_pacific_islander",
        "3": "black",
        "4": "native_american",
        "5": "other_mixed",
        "9": "unknown",
    },
    "Hispanic": {
        "0": "no",
        "1": "yes",
        "9": "unknown",
    },
    "bmi": {
        "1": "10-24.99",
        "2": "25-29.99",
        "3": "30-34.99",
        "4": "35_or_more",
        "9": "unknown",
    },
    "agefirst": {
        "0": "age_less_than_30",
        "1": "age_30_or_greater",
        "2": "nulliparous",
        "9": "unknown",
    },
    "nrelbc": {
        "0": "zero",
        "1": "one",
        "2": "two_or_more",
        "9": "unknown",
    },
    "brstproc": {
        "0": "no",
        "1": "yes",
        "9": "unknown",
    },
    "lastmamm": {
        "0": "negative",
        "1": "false_positive",
        "9": "unknown",
    },
    "surgmeno": {
        "0": "natural",
        "1": "surgical",
        "9": "unknown_or_not_menopausal",
    },
    "hrt": {
        "0": "no",
        "1": "yes",
        "9": "unknown_or_not_menopausal",
    },
    "invasive": {
        "0": "no",
        "1": "yes",
    },
    "cancer": {
        "0": "no",
        "1": "yes",
    },
    "training": {
        "0": "validation",
        "1": "training",
    },
}


def decode_value(column, value):
    if column == "count":
        return int(value)

    mapping = MAPS[column]

    if value not in mapping:
        return f"UNKNOWN_CODE_{value}"

    return mapping[value]


def convert_file(input_path, output_path):
    rows_written = 0

    with open(input_path, "r", encoding="utf-8") as input_file, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as output_file:

        writer = csv.writer(output_file)
        writer.writerow(COLUMNS)

        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != len(COLUMNS):
                raise ValueError(
                    f"Ошибка в строке {line_number}: ожидалось {len(COLUMNS)} полей, "
                    f"получено {len(values)}. Строка: {line}"
                )

            decoded_row = [
                decode_value(column, value)
                for column, value in zip(COLUMNS, values)
            ]

            writer.writerow(decoded_row)
            rows_written += 1

    print(f"Готово. Сохранено строк: {rows_written}")
    print(f"CSV файл: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="risk.txt", help="Путь к исходному risk.txt")
    parser.add_argument("--output", default="risk_decoded.csv", help="Путь к выходному CSV")
    args = parser.parse_args()

    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()