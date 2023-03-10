import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.logging import set_verbosity, ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_verbosity(ERROR)

# set colors for respective numbers dictated by index in lst
color_lst = [(255, 255, 255), (39, 33, 193), (16, 116, 238), (2, 220, 235), (1, 192, 125),
             (28, 130, 32), (227, 201, 169), (185, 107, 8), (185, 122, 176), (104, 27, 87)]


def init_model():
    """
    Loads and returns pre-trained Keras model.
    """
    model = load_model('centurion_zeta.h5', compile = False)
    return model


# Model used for digit recognition.
model = init_model()


def process_image(frame: np.ndarray):
    """
    Converts the input image to grayscale, applies a Gaussian blur and an adaptive threshold,
    and returns the processed image.
    """
    result = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, (9, 9), 1)
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 11, 2)
    return result


def extract_largest_contour(contours, min_contour_area=20000):
    """
    Finds and returns the largest four-side contour in the input list.
    """
    for c in sorted(contours, key = cv2.contourArea, reverse = True)[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(c) >= min_contour_area:
            return approx
    return None


def extract_corners(pts):
    """
    Gets the coordinates of the corners from the input points.
    """
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 1, 2), dtype = np.int32)
 
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return np.float32(rect)


def crop_image(frame, scale=1.0):
    """
    Crop the image by a certain scale.
    """
    shape = np.array(frame.shape[::-1])
    center = shape / 2
    offset = scale * shape / 2
    l_x, t_y = np.subtract(center, offset).astype(int)
    r_x, b_y = np.add(center, offset).astype(int)
    crop = frame[t_y: b_y, l_x: r_x]
    return crop


def process_square(square, idx):
    """
    Process the input image to prepare for model prediction.
    """
    _, square = cv2.threshold(square, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    square = crop_image(square, 0.8)
    square = cv2.resize(np.asarray(square), (28, 28))
    square = np.invert(square) / 255
    square = np.roll(square, -(idx // 3), axis=0)
    square = square.reshape(-1, 28, 28, 1)
    return square


def extract_squares(board):
    """
    Divides the square board into 9 by 9 squares and processes each square image.
    """
    squares = list()
    row_idx = -1
    for row in np.vsplit(board, 9):
        row_idx += 1
        for square in np.hsplit(row, 9):
            squares.append(process_square(square, row_idx))
    return np.vstack(squares)


def predict_squares(squares, min_confidence=0.65):
    """
    Predicts every square in the Sudoku puzzle.
    """
    predictions = [np.argmax(prediction) if np.amax(prediction) > min_confidence \
                       else 0 for prediction in model.predict(squares)]
    return np.asarray(predictions)


def warp(image, src, dst, dsize):
    """
    Applies a geometric transformation on the input image.
    """
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, dsize)


def extract_board(frame, largest_contour):
    """
    Given the largest contour in the frame, extract the corners of the board and the cropped board.
    """
    (height, width, _) = frame.shape
    dimensions = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
    corners = extract_corners(largest_contour)
    color_board = warp(frame.copy(), corners, dimensions, (width, height))
    board = cv2.cvtColor(color_board, cv2.COLOR_BGR2GRAY)
    board = crop_image(board, 0.98)
    return corners, board, color_board


def add_color(img, numbers):
    """
    overlays colored circles on detected digits
    """
    sec_h, sec_w = (np.array(img.shape[:2]) / 9).astype(int)
    for x in range(9):
        for y in range(9):
            number = numbers[(y * 9) + x]
            if number != 0:
                pos = (x * sec_w + int(sec_w / 2), int((y + 0.5) * sec_h))
                cv2.circle(img, pos, 18, color_lst[number], thickness=-1)
    return img
