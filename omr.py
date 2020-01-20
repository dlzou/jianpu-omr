#!/usr/bin/env python
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import util

"""
- use height of bars instead of row
- group notes with dots and lines
- group bars for or repetition
- slurs vs. brackets: check top left corner
- also need to process key/tempo above music
- parent class TextLine
"""

class AbstractLine(ABC):

    def __init__(self, img, symbols_dict):
        self.img = img
        self.symbols_dict = symbols_dict
        self._classify()

    @abstractmethod
    def _classify(self):
        pass

    def _group(self):
        pass

    def visualize(self):
        heights = []
        for (x, y, w, h), symbol in self.symbols_dict.items():
            heights.append(h)
        plt.hist(heights, bins=range(min(heights), max(heights)+1))
        plt.show()

    @staticmethod
    def construct(img, symbols_dict):
        keys = list(symbols_dict.keys())
        keys.sort(key=lambda k: k[3])  # sort by h

        if all((lambda w, h: h / w > 4)(w, h) for x, y, w, h in keys[-3:]):
            return JianPuLine(img, symbols_dict)
        else:
            return TextLine(img, symbols_dict)


class JianPuLine(AbstractLine):
    def __init__(self, img, symbols_dict):
        super().__init__(img, symbols_dict)


    def _classify(self):
        """Classify segmented symbols"""

        dim = (self.img.shape[1], self.img.shape[0])
        self.notes, self.chars, self.bars, self.dots, self.lines, self.slurs = {}, {}, [], [], [], []

        for (x, y, w, h), symbol in self.symbols_dict.items():
            if h > dim[1] / 2 and h / w > 4:
                # double bar needs to be grouped
                self.bars.append((x, y, w, h))
            elif w > dim[1] / 5 and w / h > 2:
                if y < dim[1] / 3:
                    # could be repetition brackets
                    self.slurs.append((x, y, w, h))
                elif y > dim[1] * 2/3:
                    # some dots stuck to lines
                    self.lines.append((x, y, w, h))
                else:
                    self.chars[(x, y, w, h)] = 'u'
            elif w * h > (dim[1] / 5) ** 2:
                # note or char
                self.notes[(x, y, w, h)] = '1'
            elif w * h > 20 and util.similar(w, h, ratio=0.7):
                self.dots.append((x, y, w, h))


    def __str__(self):
        return (f'Notes: {len(self.notes)}\n'
                f'Chars: {len(self.chars)}\n'
                f'Bars: {len(self.bars)}\n'
                f'Dots: {len(self.dots)}\n'
                f'Lines: {len(self.lines)}\n'
                f'Slurs: {len(self.slurs)}\n')


class TextLine(AbstractLine):
    def __init__(self, img, symbols_dict):
        super().__init__(img, symbols_dict)


    def _classify(self):
        pass


def jianpu_to_midi(img_path):
    original = cv.imread(img_path)
    assert original is not None, 'img_path does not exist'

    roi = util.page_detect_contour(original)
    assert roi is not None, 'page does not exist'
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # blurred = cv.GaussianBlur(roi, (3, 3), 0)
    adjusted = util.bleach_shadows(roi)

    # binarized = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 8)
    _, binarized = cv.threshold(adjusted, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binarized = cv.bitwise_not(binarized)
    row_imgs, row_binaries, row_ranges = util.dissect_rows(adjusted, binarized)

    # symbols_dict = util.dissect_symbols(row_imgs[3], row_binaries[3])
    # line = AbstractLine.construct(row_imgs[3], symbols_dict)
    # line = JianPuLine(row_imgs[3], symbols_dict)
    # line.visualize()
    # print(line)

    lines = []
    for img, binary in zip(row_imgs, row_binaries):
        symbols_dict = util.dissect_symbols(img, binary)
        line = AbstractLine.construct(img, symbols_dict)
        lines.append(line)
        print(line)

    util.display('Original', original)
    util.display('Binarized', np.hstack((adjusted, binarized)))
    util.display('Rows', np.hstack((util.bordered_stack(row_imgs, 0), util.bordered_stack(row_binaries, 0))))

    cv.waitKey(0)
    cv.destroyAllWindows()

    return binarized

if __name__ == '__main__':
    jianpu_to_midi('/home/dlzou/code/projects/omr/media/uploaded_img/IMG_3341.png')
