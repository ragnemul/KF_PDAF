#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "Filtro_color.hpp"


using namespace cv;
using namespace std;

Mat Filtro_color(Mat img) {
    Mat hsv;
    Mat Filtrada1;
    Mat Filtrada2;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    inRange(hsv, Scalar(0, 40, 50), Scalar(10, 255, 255), Filtrada1);
    inRange(hsv, Scalar(170, 40, 50), Scalar(180, 255, 255), Filtrada2);

    return (Filtrada1 + Filtrada2);    
}