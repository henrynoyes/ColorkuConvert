# ColorkuConvert

## Description

Converts Sudoku boards into Colorku boards. Uses a Tensorflow Keras model to perform Optical Character Recognition (OCR) on the digits of a Sudoku board. The model was generated using data from [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [TMNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist), and custom sources. The Sudoku board image is processed using methods from OpenCV. The output Colorku board images are saved to the Colorkus directory.

## Dependencies

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://github.com/opencv/opencv-python)
- [Numpy](https://numpy.org)

## Scripts

### `convert_colorku.py [-h] [-s] [-d] [-a] filename`

Converts a specified Sudoku board image into a Colorku board image.

### `group_convert_colorku.py [-h] [-s] [-d] [-a]`

Converts all Sudoku boards placed in the Sudokus directory into Colorkus, first checking if the corresponding Colorkus already exist.

### Arguments

- `filename`: filename of sudoku board image in Sudokus/, ex: daily_sudoku_1.jpg
- `-h, --help`: display help message
- `-s, --solve`: include Colorku board solution
- `-d, --display`:  display Colorku board(s) for 5s instead of storing to Colorkus/
- `-a, --array`: print array of detected digits

## Usage

1. Screenshot Sudoku board(s) and place them in the [Sudokus](https://github.com/henrynoyes/ColorkuConvert/tree/main/Sudokus) folder
2. Open a terminal in the ColorkuConvert folder (right-click folder -> New Terminal at Folder)
3. Type `python3 group_convert_colorku.py -s` to convert all Sudokus into Colorkus with solution boards

## References

See [Sudoku](https://github.com/victor-hugo-dc/Sudoku) by **victor-hugo-dc** and [OpenCV-Sudoku-Solver](https://github.com/murtazahassan/OpenCV-Sudoku-Solver) by **murtazahassan** for the inspiration and jumpstart to this project.
