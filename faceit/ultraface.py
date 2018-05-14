from argparse import ArgumentParser
import os


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/umdfaces_batch3/')
    a.add_argument('--out_all_keypoints', type=str,
                   default='data/ultraface_all_keypoints.jpg')
    return a.parse_args()


def main(flags):
    f = os.path.join(flags.in_dir, 'umdfaces_batch3_ultraface.csv')
    f = open(f)
    keys = next(f).strip().split(',')
    for line in f:
        values = next(f).strip().split(',')
        for i in range(len(keys)):
            k = keys[i]
            v = values[i]
            try:
                s = '%8.3f' % float(v)
            except:
                s = '%8s' % v
            print('%3d %11s %s' % (i, k, s))
        break


if __name__ == '__main__':
    main(parse_flags())
