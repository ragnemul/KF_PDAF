/* 
 * File:   pdaf.h
 * Author: menendez
 *
 * Created on April 6, 2013, 12:30 PM
 */

#ifndef PDAF_H
#define PDAF_H

#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#define MAXX 400
#define MAXY 300


// Define Colors in RGB
#define CYAN Scalar(0, 255, 255)
#define WHITE Scalar(255,255,255)
#define BLACK Scalar(0,0,0)
#define BLUE Scalar(255,0,0)
#define GREEN Scalar(0,255,0)
#define RED Scalar(0,0,255)
#define YELLOW Scalar(0,255,255)
#define MAGENTA Scalar (255,0,255)



#define RADIUS 3

#define CIRCLE(IMG,P,COLOR) circle(IMG, Point(P.at<float>(0, 0), P.at<float>(1, 0)), RADIUS, COLOR, 0); \
        line(img, Point(P.at<float>(0, 0), P.at<float>(1, 0) - RADIUS), \
                Point(P.at<float>(0, 0), P.at<float>(1, 0) + RADIUS), COLOR); \
        line(img, Point(P.at<float>(0, 0) - RADIUS, P.at<float>(1, 0)), \
                Point(P.at<float>(0, 0) + RADIUS, P.at<float>(1, 0)), COLOR)

// dimensions of the window
#define MAXPOINTS       5000

// Validation Gate and error_ellipse scalation factor with probability of 0.9999
#define C2              18.4
#define GAMMA           C2
#define SQRT_C          4.289522118
// #define SQRT_C          10.6
#define PG              0.9999         // gate probability

#define NULL_Bi         (float) 9999



namespace pdaf {
    using namespace cv;
    using namespace std;

    class pdaf {
    public:

        typedef struct {
            Mat Z[MAXPOINTS + 1]; // maximum number of points that can be detected
            int assigned[MAXPOINTS + 1];
            int elements;
            float Bi[MAXPOINTS + 1];
        } tvPoints;

        typedef enum tStatus {
            ACTIVE = 2, CANDIDATE = 1, FREE = 0
        } tStatus;

        typedef enum tMovButton {
            STOP = 0, UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4, UPLEFT = 5, UPRIGHT = 6, DOWNRIGHT = 7, DOWNLEFT = 8
        } tMovButton;

        tMovButton movButton;
        tStatus status;

        Scalar track_color;

        int num_assigns; // num of cycles with measurement assignment
        int num_missing; // num of cycles with measurement missing




        // Variables for JPDAF
#define MAX_EVENTS      500      // Maximun number of association events in JPDAF 
#define MAX_VALIDATED_MEASUREMENTS 100
#define MAX_TRACKS      10

        typedef struct {
            Mat M[MAX_EVENTS + 1]; // Validation matrices for each association event: values are 0, 1
            // rows are total number of unique measurements inside all the validation gates
            // cols are total number of tracks plus the t0 located in the 0 index            


            Mat valPoints[MAX_VALIDATED_MEASUREMENTS][MAX_TRACKS]; // 100 validated measurements x 10 tracks;
            // The valid dimension of the array will be the same of M dimensions
            // Stores each validated point corresponding to each validated measurement in M[0] matrix (val_matrix)
            // if M[j,t]==1 -> valPoints[j,t] has the validated point

            Mat Bjt_val[MAX_VALIDATED_MEASUREMENTS][MAX_TRACKS];
            // contains the value (Mat 1x1 of float) of Bjt or NULL_Bi is no value has been calculared previosly.


            Mat P_Chi[MAX_EVENTS + 1]; // Stores the prob of each event.
            Mat c; // normalization constant (1x1 matrix). SUM(P_Chi) for all the events must to be 1
            int events; // number of association events

            Mat Mk[MAXPOINTS]; // unique validated measurements for the two tracks we are considering
            int measurements; // number of unique validated measurements for the two tracks we are considering

            struct trackindex {
                int idx[MAX_TRACKS]; // correspondence between tracks (in order) and tracks.KF[t]). 
                // track_idx[0] = number of first track active in tracks.KF[t]
                // track_idx[1] = number of second track active in tracks.KF[t]
                int num_tracks; // num of actived tracks
            } track_index;
        } tValMatrix;

        tValMatrix valMatrix; // The validation matrix structure for JPDAF

        pdaf();
        void reset_pdaf();
        pdaf(const pdaf& orig);
        virtual ~pdaf();


        void set_status_active();
        void set_status_candidate();
        void set_status_free();

        bool status_free();
        bool status_candidate();
        bool status_active();

        void print_matrix(const char *title, CvMat *m);
        Point calc_step(Point c, Point *step, int maxX, int maxY, int *steps, int maxSteps);

        void eigen_SVD(CvMat *mat);
        void eigen_eigen(CvMat *mat);
        void eigen_eigenVV(CvMat *mat);

        void error_ellipse(Mat C, float p, CvSize *axes, int *angle, bool debug);
        void error_ellipse(CvSize *axes, int *angle, bool debug);
        Mat statePrediction(Mat F, Mat Xk);
        void statePrediction();
        Mat errorCovPrediction(Mat F, Mat Pk, Mat Q);
        void errorCovPrediction();
        Mat measurementPrediction(Mat H, Mat pXk);
        void measurementPrediction();
        Mat InnovCovariance(Mat pPk, Mat H, Mat R);
        void InnovCovariance(Mat R);
        Mat KalmanGain(Mat pPk, Mat H, Mat S);
        void KalmanGain(Mat Delta);
        Mat updateEstimation(Mat pXk, Mat Kk, Mat Zk, Mat pZk);
        Mat updateEstimation();
        Mat updateEstimation(tvPoints Zk, float PD);
        Mat updateEstimation(Mat pXk, Mat Kk, tvPoints Zk, Mat pZk, Mat Sk, float PD);
        Mat updateEstimationErrorCov(Mat Kk, Mat H, Mat pPk);
        void updateEstimationErrorCov();
        Mat updateEstimationErrorCov(Mat pPk, Mat Kk, Mat Sk, tvPoints Zk, Mat pZk, float PD);
        Mat updateEstimationErrorCov(tvPoints Zk, float PD);
        Mat updateProcessNoiseCov(Mat pXk, Mat Xk, Mat Q, int _sigma_a);
        void updateProcessNoiseCov(int _sigma_a, bool fixedQ);
        void init_points(int points);
        void init_point(vector<Point2i>);
        void init_detected_points();
        void add_detected_point(Point p);
        bool detected_points();

        float Mahalanois(Mat Zk);
        void ValidationGate(bool &create_track, bool &remove_track, int track_idx, bool shared_points);
        void ValidationGate(tvPoints clutter, tvPoints *Mk, Mat pZk, Mat Sk);
        void ValidationGate(Mat clutter[], int elements);
        float ValidationGateVolume(float Gamma, Mat Sk);
        float ValidationGateVolume(Mat Sk);
        Mat Li(tvPoints Zk, Mat pZk, Mat Sk, float PD, int i);
        Mat sumLi(tvPoints Zk, Mat pZk, Mat Sk, float PD);
        Mat Bi(tvPoints Zk, Mat pZk, Mat Sk, float PD, int i, bool B0);
        Mat innovation(Mat Zk, Mat pZk);
        Mat innovation(tvPoints Zk, Mat pZk, Mat Sk, float PD);
        Mat innovation(tvPoints Zk, float PD);

        void onMouse(int event, int x, int y, int flags, void* param);
        void paint_clutter(Mat img);
        void paint_validation_gate_points(Mat img);
        void paint_validation_gate_points(Mat img, Mat clutter[], int elements);
        void print_validation_gate(tvPoints Zk);
        void init_Bjt_values(tValMatrix *valMatrix);

        void set_stop_button();
        void set_up_button();
        void set_down_button();
        void set_left_button();
        void set_right_button();
        void set_upright_button();
        void set_downright_button();
        void set_upleft_button();
        void set_downleft_button();
        void set_home_button();
        bool up_button();
        bool down_button();
        bool stop_button();
        bool left_button();
        bool right_button();
        bool upright_button();
        bool upleft_button();
        bool downright_button();
        bool downleft_button();



        Mat img;
        bool visible;
        bool move;
        RNG posX, posY;

        RNG stepX, stepY; // random direction
        RNG maxSteps; // random steps of movement
        int steps; // number of steps that has been taken
        Point step; // single random step


        Point c; // Point that's moving in the simulator


        float Qrho;

        // control variables        

        int _scale;
        int _step;
        int _parado;




        struct timeval tNow, tPrev, tExec, tAct;
        float At;
        double ts;

        bool moving;

        // holds the number of measurements detected inside the validation gate.

        tvPoints clutter;


        clock_t currentTime, oldTime;

        double fps;


        // Kalman filter ecuations without control
        // Prediction Phase (Predict)
        Mat pXk; // statePre: project the state ahead
        // predicted state estimate: x'(k)=F*x(k-1)                                

        Mat pPk; // errorCovPre: project the error covariance ahead
        // predicted estimate covariance: P'(k)=F*P(k-1)*Ft + Q
        Mat pPk_aux;

        Mat pMk; // measurement vector (x,y); // predictedMeasurement;
        // m'(k)=H*x'(k)

        // Correction phase (Update)
        Mat Yk; // Innovation or measurement residual
        // y(k) = z(k)-H*x'(k)

        Mat Sk; // Innovation (or residual) covariance
        // S(k) = H*P'(k)Ht+R

        // extreme points of the ellipse
        Mat Sk_A;
        Mat Sk_B;
        Mat Sk_C;
        Mat Sk_D;



        Mat Kk; // gain: Kalman gain matrix: K(k)=P'(k)*Ht*inv(S(k))
        Mat Kk_aux;

        Mat Xk; // statePost: update the estimation with measurements mk     
        // corrected state: x(k)=x'(k)+K(k)*y(k)


        Mat Pk; // errorCovPost: update the estimation error covariance matrix 
        // posteriori error estimate covariance matrix: P(k)=(I-K(k)*H)*P'(k)
        Mat Pk_aux;


        Mat F; // state transition matrix
        Mat H; // measurement matrix (H)
        Mat Q; // process noise covariance matrix (Q)
        Mat Q_aux;

        Mat R; // measurement noise covariance matrix (R)

        Mat Delta; // if object is visible is identity, otherwise is the null matrix.

        tvPoints Mk; // validated measurements       

        Mat Zk; // measurement (x,y)
        Mat pZk; // predicted measurement (x,y) z'(k) = H*x'(k)

        CvSize axes; // error ellipse axes
        int angle; // angle of the error ellipse

        // Initialization of the Kalman Filter variables

        int stepsover;
        bool evalKalman;




        float C0; //C0: probability of false measurements
        float Pd; //Pd: probability of detection target


        Mat Ft_j(tValMatrix *valMatrix, tvPoints Zk, Mat pZk, Mat Sk, float PD, int j);
        Mat Bjt(tvPoints Zk, Mat pZk, Mat Sk, tValMatrix *valMatrix, int j, int t);
        float B0t(tvPoints Zk, Mat pZk, Mat Sk, tValMatrix *valMatrix, int t);
        Mat PChiZk(tValMatrix *valMatrix, tvPoints Zk, Mat pZk, Mat Sk, Mat Chi);
        void init_event_matrix(int measurements, int tracks);
        void add_event_matrix(Mat Chi);
        int tau(int j, int T, Mat Chi);
        int delta(int t, int m, Mat Chi);
        int num_false_measurements(int t, int m, Mat Chi);

        Mat updateEstimation_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t);
        Mat innovation_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t);
        Mat updateEstimationErrorCov_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t);

        void reset_valMatrix();
        void reset_track_index();
        void add_track_index(int t);
        int get_track_index(int t);
        int num_tracks();
        void set_track_idx(int idx, int value);
        void decrease_num_tracks();
        int get_track_number(int idx);
        void printEventMatrix();
        void set_valMatrix_value(int M_idx, int j, int t, float value);
        int get_num_cols_event_matrix();
        float get_valMatrix_element(int M_idx, int j, int t);
        void insert_valPoint(int j, int track_idx, Mat M);






        int locate_validated_point(Mat p);
        void add_validated_point(Mat p);
        int num_validated_measurements();
        void printEventMatrix(tValMatrix *valMatrix);
        Mat get_valPoint(tValMatrix *valMatrix, int j);
        int get_measurement_index(tValMatrix *valMatrix, Mat M);

    private:



    };

}

#endif /* PDAF_H */

