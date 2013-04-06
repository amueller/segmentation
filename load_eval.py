import sys

from pystruct.utils import SaveLogger

from msrc_first_try import eval_on_pixels, load_data


def main():
    print("loading %s ..." % sys.argv[1])
    ssvm = SaveLogger(file_name=sys.argv[1]).load()
    print(ssvm)
    data_train = load_data()


if __name__ == "__main__":
    main()
