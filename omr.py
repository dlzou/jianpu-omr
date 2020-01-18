#!/usr/bin/env python
import numpy as np
import cv2 as cv
import util


class JianPuLine:
    def __init__(self, img, symbols_dict):
        self.img = img
        self._classify(symbols_dict, (img.shape[1], img.shape[0]))

    def _classify(self, symbols_dict, dim):
        """
        Classify segmented symbols.
        ARGS:
        symbols_dict -- dict where keys are (x, y, w, h), and values are segmented symbol images
        dim -- tuple cotaining width and height of row image
        """
        self.notes, self.chars, self.bars, self.dots, self.lines, self.slurs = {}, {}, [], [], [], []
        for (x, y, w, h), symbol in symbols_dict.items():
            if h > dim[1] / 2 and h / w > 4:
                self.bars.append((x, y, w, h))
            elif w > dim[1] / 4 and w / h > 2:
                if y < dim[1] / 3:
                    self.slurs.append((x, y, w, h))
                elif y > dim[1] * 2/3:
                    # some dots stuck to lines
                    self.lines.append((x, y, w, h))
                else:
                    self.chars[(x, y, w, h)] = 'u'
            elif w * h > (dim[1] / 5) ** 2:
                # note or char
                self.notes[(x, y, w, h)] = '1'
            elif w * h > 20:
                self.dots.append((x, y, w, h))
            else:
                self.chars[(x, y, w, h)] = 'u'
        pass

    def __str__(self):
        return (f'Notes: {len(self.notes)}\n'
                f'Chars: {len(self.chars)}\n'
                f'Bars: {len(self.bars)}\n'
                f'Dots: {len(self.dots)}\n'
                f'Lines: {len(self.lines)}\n'
                f'Slurs: {len(self.slurs)}\n')


def jianpu_to_midi(img_path):
    original = cv.imread(img_path)
    assert original is not None, 'img_path does not exist'

    roi = util.page_detection_contour(original)
    assert roi is not None, 'page does not exist'
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # blurred = cv.GaussianBlur(roi, (3, 3), 0)
    adjusted = util.bleach_shadows(roi)

    # binarized = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 8)
    _, binarized = cv.threshold(adjusted, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binarized = cv.bitwise_not(binarized)
    row_imgs, row_bins, row_ranges = util.dissect_rows(adjusted, binarized)

    # single_bin = row_bins[7]
    # mask = np.zeros(tuple(s+2 for s in single_bin.shape), np.uint8)
    # area, single_bin, mask, (x, y, w, h) = cv.floodFill(single_bin, mask, (432, 30), (127), (0), (0), flags=(8 | 255 << 8))
    # util.fill_symbol(row_bins[7], row_imgs[7], (310, 30))
    symbols_dict = util.dissect_symbols(row_imgs[4], row_bins[4])
    line = JianPuLine(row_imgs[4], symbols_dict)
    print(line)

    util.display('Original', original)
    util.display('Binarized', np.hstack((adjusted, binarized)))
    util.display('Rows', np.hstack((util.bordered_stack(row_imgs, 0), util.bordered_stack(row_bins, 0))))

    cv.waitKey(0)
    cv.destroyAllWindows()

    return binarized

if __name__ == '__main__':
    jianpu_to_midi('/home/dlzou/code/projects/omr/media/uploaded_img/IMG_3350.jpg')
