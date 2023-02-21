import numpy as np
import cv2
from functions import process_image, extract_largest_contour, extract_board, extract_squares, \
    predict_squares, add_color
from sudoku import solve

if __name__ == '__main__':

    filename = 'daily2.jpg'
    display = True
    arry = True

    height = width = 450  # dimensions of the frames

    img = cv2.imread(f'Sudokus/{filename}')
    img = cv2.resize(img, (width, height))
    contours, _ = cv2.findContours(process_image(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = extract_largest_contour(contours)  # assume the largest square contour is the board

    if largest_contour is not None:
        corners, board, color_board = extract_board(img, largest_contour)
        extracted_squares = extract_squares(board)  # list of each square (image)
        predicted_squares = predict_squares(extracted_squares)  # list of predicted values
        if arry:
            print(predicted_squares)
        predicted_squares_str = np.array2string(predicted_squares, max_line_width=85, separator='').strip('[]')
        solved = solve(predicted_squares_str)

        if solved:
            color_img = add_color(color_board, predicted_squares)
            if display:
                cv2.imshow('Colorku Board', color_img)
                cv2.waitKey(5000)
            else:
                path = f'Colorkus/colored_{filename}'
                cv2.imwrite(path, color_img)
                print(f'\nComplete! Image stored in {path}')
        else:
            print(predicted_squares)
            print('\nno valid solution, check array for errors')
