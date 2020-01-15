#!/usr/bin/env python
import numpy as np
import cv2 as cv


def display(title, img):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, img)

def bordered_stack(imgs, axis):
    assert axis == 0 or axis == 1, 'axis must be 0 or 1'
    i = 1
    if axis == 0:
        while i < len(imgs):
            imgs.insert(i, np.zeros((3, imgs[i].shape[1]), np.uint8))
            i += 2
        return np.vstack(imgs)
    else:
        while i < len(imgs):
            imgs.insert(i, np.zeros((3, imgs[i].shape[0]), np.uint8))
            i += 2
        return np.hstack(imgs)


def approx(a, b, proportion=0.95):
    small, big = min(abs(a), abs(b)), max(abs(a), abs(b))
    return small / big >= proportion


def order_rect_points(points):
    rect = np.zeros((4, 2), np.float32)

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    d = np.diff(points, axis=1)
    rect[1] = points[np.argmin(d)]
    rect[3] = points[np.argmax(d)]

    # rect = [tl, tr, br, bl]
    return rect


def four_point_transform(img, points):
    assert points.shape == (4, 2), 'points must have dimension of (4, 2)'

    rect = order_rect_points(points)
    tl, tr, br, bl = rect

    max_width = int(max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br)))
    max_height = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))

    dst = np.array([
        [0, 0],
        [max_width-1, 0],
        [max_width-1, max_height-1],
        [0, max_height-1]], np.float32)

    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(img, M, (max_width, max_height))


def page_detection_contour(img, k_blur=15):
    # img = cv.imread('/home/dlzou/code/projects/omr/media/uploaded_img/IMG_3341.png')

    blurred = cv.medianBlur(img, k_blur)
    edges = cv.dilate(cv.Canny(blurred, 30, 100), np.ones((3, 3), np.uint8))
    # frame_edges = [
    #     edges[0, :], # top
    #     edges[:,0], # left
    #     edges[edges.shape[0]-1, :], # bottom
    #     edges[:, edges.shape[1]-1]] # right

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    page_contour = None
    for c in contours:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*perimeter, True)
        if len(approx) == 4:
            page_contour = approx
            break

    if page_contour is not None:
        warped = four_point_transform(img, page_contour.reshape(4, 2))
        return warped

    #     cv.drawContours(img, [page_contour], 0, (0, 255, 0), 3)
    #     display('Warped', warped)

    # blurred[edges > 0] = (0, 255, 0)
    # display('Original', np.hstack((img, blurred)))

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print('cvutil.page_detection_contour(): failed to detect page')
    return None


def page_detection_line(img, k_blur=25):
    img = cv.imread('/home/dlzou/code/projects/omr/media/uploaded_img/IMG_3350.jpg')

    blurred = cv.medianBlur(img, k_blur)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.dilate(cv.Canny(gray, 30, 150), np.ones((3, 3), np.uint8))

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=100, minLineLength=300, maxLineGap=50)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # calculate average lines
    # calculate line intersections that form largest area
    # k-mean clustering with 4 centroids

    blurred[edges > 0] = (0, 255, 0)
    display('Original', np.hstack((img, blurred)))

    cv.waitKey(0)
    cv.destroyAllWindows()


# white image adjustment
def bleach_shadows(img):
    assert img.ndim == 2, 'img must be grayscale'

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
    adjusted = blurred / closed
    return np.uint8(cv.normalize(adjusted, adjusted, 0, 255, cv.NORM_MINMAX))


def dissect_rows(bin, img, low_bound=5, min_height=20):
    assert bin.ndim == 2, 'bin must be negative binary'
    assert bin.shape[:2] == img.shape[:2], 'bin and img must have same height and width'

    # row_chart = np.zeros(bin.shape, np.uint8)
    projection_y = bin.sum(axis=1) / 255
    ranges = []
    top = -1

    for i, size in enumerate(projection_y):
        size = int(size)
        if top == -1:
            if size > low_bound:
                top = i
        elif size <= low_bound and i-top >= min_height:
            ranges.append((top, i))
            top = -1
        # row_chart[i, :size] += 255

    # col_charts = []
    # for top, bottom in ranges:
    #     cropped_bin = bin[top:bottom, :bin.shape[1]]
    #     projection_x = cropped_bin.sum(axis=0) / 255

    #     col_chart = np.zeros((bottom-top, bin.shape[1]), np.uint8)
    #     for i, size in enumerate(projection_x):
    #         size = int(size)
    #         col_chart[col_chart.shape[0]-size:, i] += 255
    #     col_charts.append(col_chart)

    row_imgs = []
    for top, bottom in ranges:
        row_imgs.append(img[top:bottom, :img.shape[1]])

    # display('Row Chart', row_chart)
    # display('Column Chart', bordered_stack(col_charts, 0))
    return row_imgs, ranges


def floodfill_size(bin, seed):
    assert isinstance(seed, tuple), 'seed must be tuple (x, y))'

    def _helper(bin, pos, w, h):
        pass

    return _helper(bin, seed, 0, 0)


if __name__ == '__main__':
    page_detection_contour(1)
