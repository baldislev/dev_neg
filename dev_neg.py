import rawpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


def show_img(img):
    plt.imshow(img)
    plt.show()


def normalize_raw(raw):
    image = np.array(raw.raw_image, dtype=np.double)
    # subtract black levels and normalize to interval [0..1]
    black = np.reshape(np.array(raw.black_level_per_channel, dtype=np.double), (2, 2))
    black = np.tile(black, (image.shape[0]//2, image.shape[1]//2))
    image = (image - black) / (raw.white_level - black)
    return image

def normalize_to_base(image, white_level, base=8):
    base = 2**base - 1
    normalized_image = base * (image / float(white_level))
    return normalized_image


def invert(negative, white_level):
    positive = white_level - negative
    return positive


def get_mosaic_indexes(color_desc, raw_pattern):
    # find the positions of the three (red, green and blue) or four base colors within the Bayer pattern
    colors = np.frombuffer(color_desc, dtype=np.byte)
    pattern = np.array(raw_pattern)
    index_0 = np.where(colors[pattern] == colors[0])
    index_1 = np.where(colors[pattern] == colors[1])
    index_2 = np.where(colors[pattern] == colors[2])
    index_3 = np.where(colors[pattern] == colors[3])
    return (index_0, index_1, index_2, index_3)


def wb_raw(image, n_colors, indexes, wb_c, white_level):
    # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is ususally the coefficient for green
    index_0, index_1, index_2, index_3 = indexes
    wb = np.zeros((2, 2), dtype=np.double) 
    wb[index_0] = wb_c[0] / wb_c[1]
    wb[index_1] = wb_c[1] / wb_c[1]
    wb[index_2] = wb_c[2] / wb_c[1]
    if n_colors == 4:
        wb[index_3] = wb_c[3] / wb_c[1]
    wb = np.tile(wb, (image.shape[0]//2, image.shape[1]//2))
    image_wb = np.array(np.clip(image * wb, 0, white_level), dtype=np.uint16)
    return image_wb


def linear_stretching(input, lower_stretch_from, upper_stretch_from, lower_stretch_to, upper_stretch_to):
    output = (input - lower_stretch_from) * ((upper_stretch_to - lower_stretch_to) / (upper_stretch_from - lower_stretch_from)) + lower_stretch_to
    return output


def gamma_correction(img, gamma=0.5, base=1):
    invGamma = 1.0 / gamma
    if base == 1:
        return img ** invGamma
    elif base == 8:
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(np.array(img, dtype=np.uint8), table)


def adjust_saturation(img, saturation=1.):
    imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s*saturation
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h,s,v])
    return cv2.cvtColor(imghsv.astype('uint8'), cv2.COLOR_HSV2RGB)


def linear_transform(img, alpha=1., beta=0.):
    """
    g(x,y,c) = alpha*f(x,y,c) + beta
    alpha controls contrast,
    beta controls brightness.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def demosaic_downsample(image, n_colors, indexes):
    index_0, index_1, index_2, index_3 = indexes
    # demosaic via downsampling
    image_demosaiced = np.empty((image.shape[0]//2, image.shape[1]//2, n_colors))
    if n_colors == 3:
        image_demosaiced[:, :, 0] = image[index_0[0][0]::2, index_0[1][0]::2]
        image_demosaiced[:, :, 1]  = (image[index_1[0][0]::2, index_1[1][0]::2] + image[index_1[0][1]::2, index_1[1][1]::2]) / 2
        image_demosaiced[:, :, 2]  = image[index_2[0][0]::2, index_2[1][0]::2]
    else: # n_colors == 4
        image_demosaiced[:, :, 0] = image[index_0[0][0]::2, index_0[1][0]::2]
        image_demosaiced[:, :, 1] = image[index_1[0][0]::2, index_1[1][0]::2]
        image_demosaiced[:, :, 2] = image[index_2[0][0]::2, index_2[1][0]::2]
        image_demosaiced[:, :, 3] = image[index_3[0][0]::2, index_3[1][0]::2]
    return image_demosaiced


def demosaic_cv(mosaic):
    pattern = cv2.COLOR_BayerRGGB2RGB # need a full implementation
    image_demosaiced = cv2.cvtColor(mosaic, pattern)
    return image_demosaiced


def process_raw(raw_filename, base, wb, alpha, beta, gamma, saturation, downsample):
    with rawpy.imread(raw_filename) as raw:
        white_level = raw.camera_white_level_per_channel[0]
        n_colors = raw.num_colors
        image = raw.raw_image

        indexes = get_mosaic_indexes(raw.color_desc, raw.raw_pattern)
        image = wb_raw(image, n_colors, indexes, wb, white_level=white_level)

        if downsample:
            image = demosaic_downsample(image, n_colors, indexes)
        else:
            image = demosaic_cv(image)

        image = invert(image, white_level=white_level)
        image = linear_stretching(image, np.min(image), np.max(image), 0, 2**base-1)
        image = gamma_correction(image, gamma=gamma, base=base)
        image = linear_transform(image, alpha=alpha, beta=5.)
        image = adjust_saturation(image, saturation=saturation)
        image = np.array(image, dtype=np.uint8) if base != 1 else image 
        return image


def init_parser(parser):
    parser.add_argument('--raw_filename', type=str, default='DSC_0301.NEF', help='input raw file for processing')
    parser.add_argument('--output', type=str, default='', help='output filename, if not specified - result will not be saved.')
    parser.add_argument('--base', type=int, default='8', help='base for pixel brightness in resulting image')
    parser.add_argument('--wb', type=list[float], default=[0.85, 1., 3., 1.], help='white balance correction values for negative')
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha param for linear transform - contrast')
    parser.add_argument('--beta', type=float, default=5., help='beta param for linear transform - brightness')
    parser.add_argument('--gamma', type=float, default=0.15, help='gamma param for gamma correction')
    parser.add_argument('--s', type=float, default=1.9, help='saturation param')
    parser.add_argument('--down', action='store_true', default=False, help='demosaic by downsampling')
    parser.add_argument('--show', action='store_true', default=False, help='show the resulting image')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = init_parser(parser)
    args = parser.parse_args()

    process_args = dict(raw_filename=args.raw_filename,
                        base=args.base,
                        wb = args.wb,
                        alpha=args.alpha,
                        beta=args.beta,
                        gamma=args.gamma,
                        saturation=args.s,
                        downsample = args.down)

    img = process_raw(**process_args)
    if args.show:
        show_img(img)

    if args.output != '':
        cv2.imwrite(args.output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
