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

    def __init__(self, img, obj_dict):
        self.img = img
        self.obj_dict = obj_dict
        self._categorize()

    @abstractmethod
    def _categorize(self):
        pass

    def _group(self):
        pass

    def visualize(self):
        heights = []
        for (x, y, w, h), obj in self.obj_dict.items():
            heights.append(h)
        plt.hist(heights, bins=range(min(heights), max(heights)+1))
        plt.show()

    @staticmethod
    def construct(img, obj_dict):
        keys = list(obj_dict.keys())
        keys.sort(key=lambda k: k[3])  # sort by h

        if all((lambda w, h: h / w > 4)(w, h) for x, y, w, h in keys[-3:]):
            return JianPuLine(img, obj_dict)
        else:
            return TextLine(img, obj_dict)


class JianPuLine(AbstractLine):
    def __init__(self, img, obj_dict):
        super().__init__(img, obj_dict)


    def _categorize(self):
        """Classify segmented objects"""

        keys_list = list(self.obj_dict.keys())
        heights = np.array([k[3] for k in keys_list])
        highest_break = util.kde_breaks(heights, 5)[-1]
        tallest_keys = [k for k in keys_list if k[3] > highest_break]
        self.bars = []
        for x, y, w, h in tallest_keys:
            if h / w > 4:
                self.bars.append((x, y, w, h))
                self.obj_dict.pop((x, y, w, h))

        assert len(self.bars) > 0, 'no bars found'
        bar_height = sum([h for x, y, w, h in self.bars]) / len(self.bars)
        print(bar_height)
        bar_top = sum([y for x, y, w, h in self.bars]) / len(self.bars)
        img_height = self.img.shape[0]
        self.notes = {}
        self.chars = {}
        self.overlines = {}
        self.underlines = []
        self.dashes = []
        self.dots = []
        self.unknowns = {}

        for (x, y, w, h), obj in self.obj_dict.items():
            if w > bar_height/3 and h < bar_height and w/h > 2:
                if y < img_height/3:
                    # is either slur or bracket
                    self.overlines[(x, y, w, h)] = obj
                elif y > img_height * 2/3:
                    # is underline
                    self.underlines.append((x, y, w, h))
                elif w < bar_height:
                    # is dash
                    self.dashes.append((x, y, w, h))
            elif util.in_range(w*h, (bar_height/10) ** 2, bar_height ** 2):
                if w*h > (bar_height/4) ** 2:
                    # either note or char
                    self.notes[(x, y, w, h)] = obj
                else:
                    # is dot
                    self.dots.append((x, y, w, h))
            else:
                self.unknowns[(x, y, w, h)] = obj


    def __str__(self):
        return (f'Bars: {len(self.bars)}\n'
                f'Notes: {len(self.notes)}\n'
                f'Chars: {len(self.chars)}\n'
                f'Top lines: {len(self.overlines)}\n'
                f'Bottom lines: {len(self.underlines)}\n'
                f'Dashes: {len(self.dashes)}\n'
                f'Dots: {len(self.dots)}\n'
                f'Unknowns: {len(self.unknowns)}\n')

class TextLine(AbstractLine):
    def __init__(self, img, obj_dict):
        super().__init__(img, obj_dict)


    def _categorize(self):
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

    # obj_dict = util.dissect_objects(row_imgs[3], row_binaries[3])
    # line = AbstractLine.construct(row_imgs[3], obj_dict)
    # # line.visualize()
    # print(line)

    lines = []
    for img, binary in zip(row_imgs, row_binaries):
        obj_dict = util.dissect_objects(img, binary)
        line = AbstractLine.construct(img, obj_dict)
        lines.append(line)
        print(line)

    util.display('Original', original)
    util.display('Binarized', np.hstack((adjusted, binarized)))
    util.display('Rows', np.hstack((util.bordered_stack(row_imgs, 0), util.bordered_stack(row_binaries, 0))))

    cv.waitKey(0)
    cv.destroyAllWindows()

    return binarized

if __name__ == '__main__':
    jianpu_to_midi('/home/dlzou/code/projects/omr/media/uploaded_img/IMG_3348.jpg')
