import os
import numpy as np
import cv2
import argparse
from functions import process_image, extract_largest_contour, extract_board, extract_squares, \
    predict_squares, add_color
from sudoku import solve
from pathlib import Path

parser = argparse.ArgumentParser(description='Converts Sudoku board into Colorku board.')
parser.add_argument('-s', '--solve', action='store_true', dest='inc_solu',
                    help='include Colorku board solution')
parser.add_argument('-d', '--display', action='store_true', dest='display',
                    help='display Colorku board(s) for 5s instead of storing')
parser.add_argument('-a', '--arry', action='store_true', dest='arry',
                    help='print array of detected digits')

if __name__ == '__main__':

    args = parser.parse_args()
    inc_solu = args.inc_solu
    display = args.display
    arry = args.arry

    cwd = os.getcwd()
    sud_dir = f'{cwd}\Sudokus'
    cku_dir = f'{cwd}\Colorkus'
    sud_lst = [file.name for file in Path(sud_dir).iterdir()]
    cku_lst = [file.name for file in Path(cku_dir).iterdir()]
    height = width = 450  # dimensions of the frames
    new_sud_lst = list()

    for sud in sud_lst:
        if not any(sud in cku for cku in cku_lst):
            new_sud_lst.append(sud)

    for filename in new_sud_lst:
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
            solu = solve(predicted_squares_str)

            if solu:
                color_img = add_color(color_board, predicted_squares)
                pos_arry = np.where(predicted_squares > 0, 0, 1)
                solu_squares = [*map(int, solu.values())] * pos_arry
                all_squares = predicted_squares + solu_squares
                solu_img = add_color(color_board.copy(), all_squares)

                if display:
                    cv2.imshow('Colorku Board', color_img)
                    if inc_solu:
                        cv2.imshow('Colorku Board Solution', solu_img)
                    cv2.waitKey(5000)
                else:
                    path = f'Colorkus/colored_{filename}'
                    cv2.imwrite(path, color_img)
                    print(f'\nComplete! Image stored in {path}')
                    if inc_solu:
                        solu_path = f'Colorkus/SOLU_colored_{filename}'
                        cv2.imwrite(solu_path, solu_img)
                        print(f'\nand solution stored in {solu_path}')
            else:
                print(predicted_squares)
                print('\nno valid solution, check array for errors')
