
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "num_blobs.hpp"

using namespace cv;
using namespace std;

vector<Point2i> blobs(Mat frame) {

    // Use Canny instead of threshold to catch squares with gradient shading
    Mat bw;
    Canny(frame, bw, 100, 200, 5);

    // Find contours
    vector<vector<Point> > contours;
    findContours(bw.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // get the moments
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        mu[i] = moments(contours[i], false);
    }

    // get the centroid of figures.
    vector<Point2i> mc(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        mc[i] = Point2i(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

    vector<Vec4i> hierarchy;
    // draw contours
    Mat drawing(bw.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 0, 0); // B G R values
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(drawing, mc[i], 4, color, -1, 8, 0);
    }

    // show the resultant image
    namedWindow("Contours", CV_WINDOW_KEEPRATIO);
    resizeWindow("Contours", 1920 / 3, 1080 / 3);
    imshow("Contours", drawing);
/*
    namedWindow("Image with center", CV_WINDOW_KEEPRATIO);
    resizeWindow("Image with center", 1920 / 2, 1080 / 2);
    
    namedWindow("Canny image", CV_WINDOW_KEEPRATIO);
    resizeWindow("Canny image", 1920 / 2, 1080 / 2);
   
 */
    // cout << "Hay " << contours.size() << " piezas.";
    // imshow("Image with center", frame);
    // imshow("Canny image", bw);
    // waitKey(0);
    
    return mc;
}