#include <iostream>
#include <cstdlib>
#include "opencv/cv.hpp"
#include "opencv2/imgproc.hpp"

#include "Filtro_color.hpp"
#include "deteccion_cuadrados.hpp"
#include "num_blobs.hpp"
#include "dilatacion.hpp"
#include "../../pdaf.hpp"


#define ESC 27

using namespace std;
using namespace cv;

bool isClosed(Point2i p1, Point2i p2, Point2i p) {
    int y;

    y = (int) (((p.x - p1.x)*(p2.y - p1.y)) / (p2.x - p1.x) + p1.y);

    // cout << " {y: " << y << " p.y: " << p.y << "} ";

    return (p.y > y ? true : false);
}

void show_status(Mat img, Point2i p1, Point2i p2, Point2i p) {
    if (isClosed(p1, p2, p)) {
        putText(img, "CERRADA", Point(img.cols - 200, 30), FONT_HERSHEY_DUPLEX, 1.0, GREEN, 2);
        line(img, p1, p2, GREEN, 2);
    } 
    else {
        putText(img, "ABIERTA", Point(img.cols - 200, 30), FONT_HERSHEY_DUPLEX, 1.0, RED, 2);
        line(img, p1, p2, RED, 2);
    }

    // cout << "p1: " << p1 << " p2: " << p2 << " p:" << p << isClosed(p1,p2,p) << endl;
}

int main(int argc, char** argv) {

    string video_file = "/media/psf/Google Drive/My Drive/UC3M/Procesamiento Imagenes/Trabajo/20191125_154236.mp4";

    VideoCapture cap;
    Mat frame;
    char keypressed = 0;
    bool success;

    cap.open(video_file);
    if (!cap.isOpened()) {
        std::cout << "Error in video capture: check pathname" << endl;
        return 1;
    }

    // create window cavas to show the image
    namedWindow("original", CV_WINDOW_KEEPRATIO);
    resizeWindow("original", 1920 / 3, 1080 / 3);

    // namedWindow("filtrado", CV_WINDOW_KEEPRATIO);
    // resizeWindow("filtrado", 1920 / 4, 1080 / 4);

    Mat img;
    Mat frame_dilatado;
    vector<Point2i> centroids;
    pdaf::pdaf pdaf_filter;


    // PDAF custom vars
    // initialized the centroid of the tracked object
    pdaf_filter.pXk.at<float>(0, 0) = 925;
    pdaf_filter.pXk.at<float>(1, 0) = 990;

    pdaf_filter.Pd = 0.02;      // 0.05
    pdaf_filter.ts = 0.15;      // 0.15
    // circle(frame, pdaf_filter.c, RADIUS, BLUE, -1);

    Point2i p1 = Point(0, 950);
    Point2i p2 = Point(1920, 1000);
    Point2i p;

    bool remove_track, activate_track;
   
    waitKey(0);

    // main loop
    while (1) {

        cap >> frame;
        if (frame.empty())
            break;
        
        Mat gray;
        cvtColor(frame, gray, CV_RGB2GRAY);
        imshow("Debug Window", gray);

        p = Point2i(pdaf_filter.pXk.at<float>(0, 0), pdaf_filter.pXk.at<float>(1, 0));
        show_status(frame, p1, p2, p);

        img = Filtro_color(frame);

        // con size de cuadro = 10*10 toma el blob cerrado
        deteccion_cuadrados(img, 10);

        img = dilatacion(img);

        centroids = blobs(img);


        // insert all the centroids in pdaf clutter
        pdaf_filter.init_point(centroids);
        // all the centroids detected are drawn
        pdaf_filter.paint_clutter(frame);

        // update process noise covariance
        pdaf_filter.updateProcessNoiseCov(10, true); // kalman_sigma_a->get_value()

        // prediction
        pdaf_filter.statePrediction(); // pXk = F * Xk;
        pdaf_filter.errorCovPrediction(); // pPk = F * Pk * F.t() + Q;                
        pdaf_filter.measurementPrediction(); // pZk = H * pXk;     

        pdaf_filter.Delta = (centroids.size() > 0 ?
                Mat::eye(2, 2, CV_32F) :
                Mat::zeros(2, 2, CV_32F));



        // correction phase
        pdaf_filter.R = Mat::eye(2, 2, CV_32F) * 10; // kalman_R->get_value()
        pdaf_filter.InnovCovariance(pdaf_filter.R); // Sk = H * pPk * H.t() + R; 
        pdaf_filter.KalmanGain(pdaf_filter.Delta); // pPk * H.t() * S.inv() * Delta;


        // association
        pdaf_filter.ValidationGate(activate_track, remove_track, 0, true);
        pdaf_filter.Xk = pdaf_filter.updateEstimation(pdaf_filter.Mk, pdaf_filter.Pd);
        // cout << "pdaf_filter.Xk:" << pdaf_filter.Xk << endl;

        pdaf_filter.Pk = pdaf_filter.updateEstimationErrorCov(pdaf_filter.Mk, pdaf_filter.Pd);
        // cout << "pdaf_filter.Pk:" << pdaf_filter.Pk << endl;


        // error ellipse
        pdaf_filter.error_ellipse(&pdaf_filter.axes, &pdaf_filter.angle, false);
        if (pdaf_filter.axes.width >= 0 && pdaf_filter.axes.height >= 0)
            ellipse(frame, Point(pdaf_filter.pXk.at<float>(0, 0), pdaf_filter.pXk.at<float>(1, 0)),
                pdaf_filter.axes, (int) (pdaf_filter.angle), 0, 360, YELLOW, 1, CV_AA, 0);
        pdaf_filter.paint_validation_gate_points(frame);


        // show the images
        imshow("original", frame);
        // imshow("filtrado", img);

        char c = (char) waitKey(25);
        if (c == ESC)
            break;
        if (c == ' ')
            waitKey(0);
    }

    // Freeing memory - not really needed because Mat is auto freed
    destroyWindow("original");
    // destroyWindow("filtrado");

    cap.release();

    return 0;
}

