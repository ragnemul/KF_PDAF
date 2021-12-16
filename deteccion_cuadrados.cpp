#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "deteccion_cuadrados.hpp"

using namespace cv;
using namespace std;

void deteccion_cuadrados(Mat BW_img, int square_size) {
    Mat Cuadrado = getStructuringElement(MORPH_RECT, Size(square_size, square_size));
    erode(BW_img, BW_img, Cuadrado);
    dilate(BW_img, BW_img, Cuadrado);
    
}