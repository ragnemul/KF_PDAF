/* 
 * File:   pdaf.cpp
 * Author: menendez
 * 
 * Created on April 6, 2013, 12:30 PM
 */
#include <opencv2/imgproc.hpp>
#include "pdaf.hpp"


using namespace std;
using namespace cv;

namespace pdaf {

    pdaf::pdaf() {
        reset_pdaf();
    }

    void pdaf::reset_pdaf() {

        status = FREE;

        num_assigns = 0; // num of cycles with measurement assignment
        num_missing = 0; // num of cycles with measurement missing

        Mat img(MAXY, MAXX, CV_8UC3);
        visible = true;
        move = true;

        posX(0xffffffff);
        posY(0xBBBBBBBB); // location of each random point
        stepX(0xffffffff);
        stepY(0xBBBBBBBB); // random direction
        maxSteps(0xDDDDDDDD); // random steps of movement
        steps = 0; // number of steps that has been taken
        step = Point(stepX.uniform(-2, 2), stepY.uniform(-2, 2)); // single random step       



        Qrho = 194;

        // control variables
        _scale = 1;
        _step = 0;
        _parado = 1;

        At = 0;
        ts = 0.04;

        moving = false;

        fps = 0;

        pMk = Mat(2, 1, CV_32F);
        Delta = Mat::eye(2, 2, CV_32F); // if we have kalman gain (I) or not (zero matrix)
        Mk.elements = 0;
        Zk = Mat(2, 1, CV_32F); // measurement (x,y)
        pZk = Mat(2, 1, CV_32F); // predicted measurement (x,y) z'(k) = H*x'(k)

        F = (Mat_<float>(4, 4) <<
                1, 0, ts, 0,
                0, 1, 0, ts,
                0, 0, 1, 0,
                0, 0, 0, 1
                );

        H = (Mat_<float>(2, 4) <<
                1, 0, 0, 0,
                0, 1, 0, 0
                );

        Q = (Mat_<float>(4, 4) <<
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, pow(Qrho*ts, 2), 0,
                0, 0, 0, pow(Qrho*ts, 2)
                );

        R = Mat::eye(2, 2, CV_32F);

        Pk = Mat::eye(4, 4, CV_32F);

        Xk = (Mat_<float>(4, 1) << 0, 0, 0, 0);
        pXk = Xk;
        //pPk = Pk;

        randn(Zk, Scalar::all(0), Scalar::all(0.1));
        randn(Xk, Scalar::all(0), Scalar::all(0.1));

        stepsover = 0;

        c = Point(rand() % MAXX, rand() % MAXY);
        movButton = STOP;
        step = Point(0, 0);

        visible = true;
        move = true;


        C0 = 0.5;
        Pd = 0.35;

    }

    pdaf::pdaf(const pdaf& orig) {
    }

    pdaf::~pdaf() {
    }

    void pdaf::set_status_active() {
        status = ACTIVE;
    }

    void pdaf::set_status_candidate() {
        status = CANDIDATE;
    }

    void pdaf::set_status_free() {
        status = FREE;
    }

    bool pdaf::status_free() {
        return ((status == FREE) ? true : false);
    }

    bool pdaf::status_candidate() {
        return ((status == CANDIDATE) ? true : false);
    }

    bool pdaf::status_active() {
        return ((status == ACTIVE) ? true : false);
    }

    void pdaf::print_matrix(const char *title, CvMat *m) {
        int r, c;
        printf("\n%s(%dx%d):(", title, m->rows, m->cols);
        for (r = 0; r < m->rows; r++) {
            printf("[");
            for (c = 0; c < m->cols; c++) {
                CvScalar scal = cvGet2D(m, r, c);
                printf("%f", scal.val[0]);
                if (c < m->cols - 1) printf(",");
            }
            printf("]");
        }
        printf(")\n");

    }

    Point pdaf::calc_step(Point c, Point *step, int maxX, int maxY, int *steps, int maxSteps) {
        int sign = rand() % 2 ? -1 : 1;


        if (*steps >= maxSteps) {
            if (c.x >= maxX - maxSteps) // [-1,0]
                step->x = rand() % 2 * (-1);
            else if (c.x <= 0 + maxSteps) // [0,1]
                step->x = rand() % 2;
            else // [-1,1]]
                step->x = rand() % 2 * sign;

            if (c.y >= maxY - maxSteps) // [-1,0]
                step->y = rand() % 2 * (-1);
            else if (c.y <= 0 + maxSteps) // [0,1]]
                step->y = rand() % 2;
            else // [-1,1]]
                step->y = rand() % 2 * sign;

            *steps = 0;
        } else
            ++*steps;


        c += *step;
        return c;

    }

    void pdaf::eigen_SVD(CvMat *mat) {

        CvMat* Ut = cvCreateMat(mat->rows, mat->cols, mat->type);
        CvMat* Dt = cvCreateMat(mat->rows, mat->cols, mat->type);
        CvMat* Vt = cvCreateMat(mat->rows, mat->cols, mat->type);
        CvMat* Resultt = cvCreateMat(mat->rows, mat->cols, mat->type);

        // print_matrix("matt", &mat);
        cvSVD(mat, Dt, Ut, Vt, mat->type); // A = U D V^T
        cvMatMul(Ut, Vt, Resultt);
        printf("EigenValues by cvSVD: ");
        for (int i = 0; i < mat->rows; i++)
            printf("%f ", cvmGet(Dt, i, i));
        printf("\n");
    }

    void pdaf::error_ellipse(Mat C, float p, CvSize *axes, int *angle, bool debug) {
        // cov_mat: kalman->error_cov_pre
        // p: confidence limit f=p/100



        float f; // f=p/100;
        float Kf; // Kf=ln(1-f);
        float r0, r1; // r0,1=sqrt(-2*Kf*Evals0,1)
        Mat A; // A=inv(C)
        Mat Evals; // Eigen values
        Mat Evects; // Eigen vectors
        // when CvMAt *cov_mat is declared: Mat C(cov_mat, true); // Covariance matrix (kalman -> error_cov_pre)


        f = p / 100;
        Kf = log(1 - f);
        invert(C, A);
        eigen(A, Evals, Evects); // eigenvalues sorted desc   
        r0 = sqrt(-2 * Kf * Evals.at<float>(0, 0));
        r1 = sqrt(-2 * Kf * Evals.at<float>(0, 1));

        *axes = cvSize(r0, r1);
        *angle = (180 / CV_PI) * atan2(Evects.at<double>(0, 1), Evects.at<double>(0, 0));

        if (debug) {
            cout << "C:" << " " << C << endl;
            // cout << "Inverse of C:" << " " << A << endl;
            // cout << "EigenValues by eigen():" << " " << Evals << endl;
            // cout << "EigenVectors by eigen():" << " " << Evects << endl;
        }
    }

    void pdaf::error_ellipse(CvSize *axes, int *angle, bool debug) {
        using namespace std;
        using namespace cv;

        float rho_xy = Sk.at<float>(0, 1);
        float rho_x2 = Sk.at<float>(0, 0);
        float rho_y2 = Sk.at<float>(1, 1);
        float s, t;


        float num = 2 * rho_xy;
        float den = (rho_x2 - rho_y2);
        float dos_theta = atan(num / den);

        // paso a grados
        *angle = dos_theta * 180 / CV_PI;

        // calculo del cuadrante de 2t
        if (num >= 0 && den >= 0)
            *angle = *angle + 0;
        if (num >= 0 && den < 0)
            *angle = *angle + 180;
        if (num < 0 && den < 0)
            *angle = *angle + 180;
        if (num < 0 && den >= 0)
            *angle = *angle + 360;

        *angle = *angle / 2;

        s = (rho_x2 + rho_y2) / 2;
        t = sqrt((pow((rho_x2 - rho_y2), 2) / 4) + pow(rho_xy, 2));

        if (debug) {
            cout << "angle: " << *angle << " s: " << s << " t: " << t << " eje mayor: " << sqrt(s + t) << " eje menor: " << sqrt(s - t) << endl;
            cout << "Sk: " << Sk << endl;
        }

        axes->width = (int) _scale * SQRT_C * sqrt(s + t);
        axes->height = (int) _scale * SQRT_C * sqrt(s - t);

        //cout << "-------" << endl;
        //cout << "Sk[t]: " << Sk;
        //cout << " SkA: " << Sk_A << " SkB: " << Sk_B <<
        //                " SkC: " << Sk_C << " SkD: " << Sk_D << endl;
        //        cout << "-------" << endl;
    }

    void pdaf::statePrediction() {
        // project the state ahead
        pXk = F * Xk;
    }

    Mat pdaf::statePrediction(Mat F, Mat Xk) {
        // project the state ahead
        return ( F * Xk);
    }

    void pdaf::errorCovPrediction() {
        // project the error covariance ahead
        pPk = F * Pk * F.t() + Q;
    }

    Mat pdaf::errorCovPrediction(Mat F, Mat Pk, Mat Q) {
        // project the error covariance ahead
        return (F * Pk * F.t() + Q);
    }

    void pdaf::measurementPrediction() {
        // project the measurement ahead
        pZk = H * pXk;
    }

    Mat pdaf::measurementPrediction(Mat H, Mat pXk) {
        // project the measurement ahead
        return (H * pXk);
    }

    void pdaf::InnovCovariance(Mat R) {
        // compute the Innovation Covariance Sk       

        Sk = H * pPk * H.t() + R;


        // calculate the extreme points of the ellipse

        float rho_xy = Sk.at<float>(0, 1);
        float rho_x2 = Sk.at<float>(0, 0);
        float rho_y2 = Sk.at<float>(1, 1);
        float p;
        float rho_x;
        float rho_y;

        rho_x = sqrt(rho_x2);
        rho_y = sqrt(rho_y2);

        p = rho_xy / (rho_x * rho_y);
        Sk_A = (Mat_<float>(2, 1) << p * SQRT_C * rho_x, SQRT_C * rho_y);
        Sk_B = (Mat_<float>(2, 1) << SQRT_C * rho_x, p * SQRT_C * rho_y);
        Sk_C = (Mat_<float>(2, 1) << p * SQRT_C * rho_x, SQRT_C * rho_y);
        Sk_D = (Mat_<float>(2, 1) << SQRT_C * rho_x, p * SQRT_C * rho_y);
    }

    Mat pdaf::InnovCovariance(Mat pPk, Mat H, Mat R) {
        // compute the Innovation Covariance Sk
        Mat S;

        S = H * pPk * H.t() + R;
        return (S);
    }

    void pdaf::KalmanGain(Mat Delta) {
        // compute the Kalman Gain    

        Kk = pPk * H.t() * Sk.inv() * Delta;
    }

    Mat pdaf::KalmanGain(Mat pPk, Mat H, Mat S) {
        // compute the Kalman Gain    

        return (pPk * H.t() * S.inv());
    }

    Mat pdaf::innovation(Mat Zk, Mat pZk) {
        return (Zk - pZk);
    }

    Mat pdaf::innovation(tvPoints Zk, Mat pZk, Mat Sk, float PD) {
        int i;
        Mat sum = Mat::zeros(pZk.rows, pZk.cols, CV_32F);
        float calcBi;
        Mat calcInnovation;

        for (i = 0; i < Zk.elements; i++) {
            if (Zk.Bi[i] == NULL_Bi) {
                Zk.Bi[i] = this->Bi(Zk, pZk, Sk, PD, i, false).at<float>(0, 0);
            }
            sum += Zk.Bi[i] * innovation(Zk.Z[i], pZk);
        }

        return (sum);
    }

    Mat pdaf::updateEstimation(Mat pXk, Mat Kk, tvPoints Zk, Mat pZk, Mat Sk, float PD) {
        // update estimate with all the measurements inside the validation gate
        // pXk: predicted estate
        // Kk: Kalman gain
        // Zk: points inside the validation gate    
        // pZk: predicted measurement
        // Sk: conv matrix of innovation
        // PD: prob. of detection      

        Yk = innovation(Zk, pZk, Sk, PD);
        return (pXk + Kk * Yk);
    }

    // PDAF combined innovation

    Mat pdaf::innovation(tvPoints Zk, float PD) {
        int i;
        Mat sum = Mat::zeros(pZk.rows, pZk.cols, CV_32F);

        for (i = 0; i < Zk.elements; i++) {
            if (Zk.Bi[i] == NULL_Bi) {
                Zk.Bi[i] = this->Bi(Zk, pZk, Sk, PD, i, false).at<float>(0, 0);
            }
            sum += Zk.Bi[i] * innovation(Zk.Z[i], pZk);
            // cout << "Zk.Bi[" << i << "]" << Zk.Bi[i] << " Innovation:" << innovation(Zk.Z[i], pZk) << endl;
        }
        Yk = sum;

        return (sum);
    }

    Mat pdaf::updateEstimation(tvPoints Zk, float PD) {
        // update estimate with all the measurements inside the validation gate
        // pXk: predicted estate
        // Kk: Kalman gain
        // Zk: points inside the validation gate    
        // pZk: predicted measurement
        // Sk: conv matrix of innovation
        // PD: prob. of detection

        Mat innov;

        innov = innovation(Zk, PD);
        return (pXk + Kk * innov);

    }

    Mat pdaf::updateEstimation() {
        // update estimate with only the real measurement Zk, without PDAF
        // pXk: predicted estate
        // Kk: Kalman gain   
        // Zk: real position of the unique point
        // pZk: predicted measurement.

        Yk = innovation(Zk, pZk);
        return (pXk + Kk * Yk);
    }

    Mat pdaf::updateEstimation(Mat pXk, Mat Kk, Mat Zk, Mat pZk) {
        // update estimate with only the real measurement Zk, without PDAF
        // pXk: predicted estate
        // Kk: Kalman gain   
        // Zk: real position of the unique point
        // pZk: predicted measurement.

        Yk = innovation(Zk, pZk);
        return (pXk + Kk * Yk);
    }

    void pdaf::updateEstimationErrorCov() {
        // update the error covariance in standard Kalman Filter        
        Mat I = Mat::eye(Kk.rows, H.cols, Kk.type());

        Pk = (I - Kk * H) * pPk;
    }

    Mat pdaf::updateEstimationErrorCov(Mat Kk, Mat H, Mat pPk) {
        // update the error covariance without PDAF
        Mat I = Mat::eye(Kk.rows, H.cols, Kk.type());

        return ( (I - Kk * H) * pPk);
    }

    Mat pdaf::updateEstimationErrorCov(Mat pPk, Mat Kk, Mat Sk, tvPoints Zk, Mat pZk, float PD) {
        // update error cov matrix of the updated state considering PDAF approach
        // pPk: predicted Cov Matrix of the estimation state
        // Kk: kalman gain
        // Sk: innovation cov matrix
        // Zk: validation gate points
        // pZk: predicted measurement

        int i;
        float B0;
        Mat Pc; // cov matrix of the stated update with the correct measurement
        Mat Ps; // spread of innovations
        Mat Vi; // innovation of each measurement inside the validation gate
        Mat Vk; // combine      d innovation
        Mat sum = Mat::zeros(2, 2, CV_32F);

        Mat Pk; // return value

        B0 = Bi(Zk, pZk, Sk, PD, 0, true).at<float>(0, 0);
        Pc = pPk - Kk * Sk * Kk.t();

        Vk = innovation(Zk, pZk, Sk, PD);
        Yk = Vk;

        for (i = 0; i < Zk.elements; i++) {
            Vi = innovation(Zk.Z[i], pZk);

            if (Zk.Bi[i] == NULL_Bi) {
                Zk.Bi[i] = Bi(Zk, pZk, Sk, PD, i, false).at<float>(0, 0);
            }
            // sum += Zk.Bi[i] * Vi * Vi.t();
            sum += Bi(Zk, pZk, Sk, PD, i, false).at<float>(0, 0) * Vi * Vi.t();

        }
        Ps = Kk * (sum - Vk * Vk.t()) * Kk.t();

        Pk = B0 * pPk + (1 - B0) * Pc + Ps;

        return (Pk);
    }

    Mat pdaf::updateEstimationErrorCov(tvPoints Zk, float PD) {
        // update error cov matrix of the updated state considering PDAF approach
        // pPk: predicted Cov Matrix of the estimation state
        // Kk: kalman gain
        // Sk: innovation cov matrix
        // Zk: validation gate points
        // pZk: predicted measurement

        int i;
        float B0;
        Mat Pc; // cov matrix of the stated update with the correct measurement
        Mat Ps; // spread of innovations
        Mat Vi; // innovation of each measurement inside the validation gate
        Mat Vk; // combine      d innovation
        Mat sum = Mat::zeros(2, 2, CV_32F);

        Mat Pk; // return value

        B0 = Bi(Zk, pZk, Sk, PD, 0, true).at<float>(0, 0);
        Pc = pPk - Kk * Sk * Kk.t();


        Vk = Yk; // from updateEstimation in the previous step

        for (i = 0; i < Zk.elements; i++) {
            Vi = innovation(Zk.Z[i], pZk);

            if (Zk.Bi[i] == NULL_Bi) {
                Zk.Bi[i] = Bi(Zk, pZk, Sk, PD, i, false).at<float>(0, 0);
            }
            sum += Zk.Bi[i] * Vi * Vi.t();
        }
        Ps = Kk * (sum - Vk * Vk.t()) * Kk.t();

        Pk = B0 * pPk + (1 - B0) * Pc + Ps;

        return (Pk);
    }

    void pdaf::updateProcessNoiseCov(int _sigma_a, bool fixedQ) {
        // update the process noise covariance matrix Q depending of values between prediction state (pXk) 
        // and correction state (Xk)
        float vx_pre, vx_post, vy_pre, vy_post;
        float ax, ay, p;
        float norm;


        vx_pre = pXk.at<float>(2, 0);
        vy_pre = pXk.at<float>(3, 0);

        vx_post = Xk.at<float>(2, 0);
        vy_post = Xk.at<float>(3, 0);

        norm = sqrt(pow(vx_post, 2) + pow(vy_post, 2));

        ax = ((vx_pre - vx_post) / ts); // este tiempo debería ser la diferencia entre el estado corregido y el estimado
        ay = ((vy_pre - vy_post) / ts);

        // cout << "(vxpre,vypre)= " << Point2f(vx_pre, vy_pre) << " (vxpost,vypost)= " << Point2f(vx_post, vy_post);
        // cout << " norm: " << sqrt(pow(vx_post, 2) + pow(vy_post, 2)) << " (ax,ay)= " << Point2f(ax, ay) << " ts: " << ts << endl;

        if (fixedQ) {
            /*
            Mat A = *(Mat_<float>(2, 2) <<
                    pow(_sigma_a, 2), 0,
                    0, pow(_sigma_a, 2));
                        

            Mat G = *(Mat_<float>(4, 2) <<
                     0, 0,
                     0, 0,
                     //  pow(ts, 2) / 2, 0,
                     //  0, pow(ts, 2) / 2,
                    ts, 0,
                    0, ts);

            Q = G * A * G.t();  
             */
            Q = _sigma_a * Mat::eye(4, 4, CV_32F);
        } else
            if (norm <= (float) _parado) { // PARADO
            /* Mat A = *(Mat_<float>(2, 2) <<
                      pow(_sigma_a, 2), 0,
                      0, pow(_sigma_a, 2));

              Mat G = *(Mat_<float>(4, 2) <<
                      // 0, 0,
                      // 0, 0,
                      pow(ts, 2) / 2, 0,
                      0, pow(ts, 2) / 2,
                      ts, 0,
                      0, ts);
              Q = G * A * G.t(); 
 
             */
            Q = _sigma_a * Mat::eye(4, 4, CV_32F);
            // cout << "PARADO" << endl;            
        } else { // only for KF
            p = 0;



            Mat A = (Mat_<float>(2, 2) <<
                    pow(ax, 2), p * ax*ay,
                    p * ay * ax, pow(ay, 2));

            Mat G = (Mat_<float>(4, 2) <<
                    0, 0,
                    0, 0,
                    //pow(ts, 2) / 2, 0,
                    //0, pow(ts, 2) / 2,
                    ts, 0,
                    0, ts);


            Q = G * A * G.t();
        }
    }

    Mat pdaf::updateProcessNoiseCov(Mat pXk, Mat Xk, Mat Q, int _sigma_a) {
        // update the process noise covariance matrix Q depending of values between prediction state (pXk) 
        // and correction state (Xk)
        float vx_pre, vx_post, vy_pre, vy_post;
        float ax, ay, p;
        float norm;
        Mat Qret;

        vx_pre = pXk.at<float>(2, 0);
        vy_pre = pXk.at<float>(3, 0);

        vx_post = Xk.at<float>(2, 0);
        vy_post = Xk.at<float>(3, 0);

        norm = sqrt(pow(vx_post, 2) + pow(vy_post, 2));

        ax = ((vx_pre - vx_post) / ts); // este tiempo debería ser la diferencia entre el estado corregido y el estimado
        ay = ((vy_pre - vy_post) / ts);


        // cout << "(vxpre,vypre)= " << Point2f(vx_pre, vy_pre) << " (vxpost,vypost)= " << Point2f(vx_post, vy_post);
        // cout << " norm: " << sqrt(pow(vx_post, 2) + pow(vy_post, 2)) << " (ax,ay)= " << Point2f(ax, ay) << " ts: " << ts << endl;


        if (norm <= (float) _parado) { // PARADO
            Qret = (Mat_<float>(4, 4) <<
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, pow(_sigma_a*ts, 2), 0,
                    0, 0, 0, pow(_sigma_a*ts, 2));
            // cout << "PARADO" << endl;
            return Qret;
        } else {
            p = 1;
            Mat A = (Mat_<float>(2, 2) <<
                    pow(ax, 2), p * ax*ay,
                    p * ay * ax, pow(ay, 2));

            Mat G = (Mat_<float>(4, 2) <<
                    // pow(ts, 2) / 2, 0,
                    // 0, pow(ts, 2) / 2,
                    0, 0,
                    0, 0,
                    ts, 0,
                    0, ts);

            Mat t_G(2, 4, CV_32F);
            transpose(G, t_G);
            Qret = G * A * G.t();
            return (Qret);
        }

    }

    // update the control parameters



    

    void pdaf::init_Bjt_values(tValMatrix *valMatrix) {
        for (int j = 0; j < MAX_VALIDATED_MEASUREMENTS; j++)
            for (int t = 0; t < MAX_TRACKS; t++)
                valMatrix -> Bjt_val[j][t] = (Mat_<float>(1, 1) << NULL_Bi);
    }

    // stores the points noise

    void pdaf::init_points(int points) {

        for (int i = 0; i < points; i++) {
            clutter.Z[i] = (Mat_<float>(2, 1) << posX.uniform(1, MAXX), posY.uniform(1, MAXY));
            clutter.Bi[i] = NULL_Bi;
            clutter.assigned[i] = -1;
        }
        clutter.elements = points;
    }
    
    void pdaf::init_point(vector<Point2i> centroids) {
        int elements = centroids.size();
        for (int i=0; i<elements; i++) {
            clutter.Z[i] = (Mat_<float>(2, 1) << centroids[i].x, centroids[i].y);
            clutter.Bi[i] = NULL_Bi;
            clutter.assigned[i] = -1;
        }
        clutter.elements = elements; 
    }

    void pdaf::init_detected_points() {
        clutter.elements = 0;
    }

    void pdaf::add_detected_point(Point p) {
        clutter.Z[clutter.elements] = (Mat_<float>(2, 1) << p.x, p.y);
        clutter.Bi[clutter.elements] = NULL_Bi;
        clutter.elements++;
    }

    bool pdaf::detected_points() {
        return (clutter.elements == 0 ? false : true);
    }

    // draws the point

    void pdaf::paint_clutter(Mat img) {

        // shows the clutter
        for (int i = 0; i < clutter.elements; i++) {
            circle(img, Point(clutter.Z[i].at<float>(0, 0), clutter.Z[i].at<float>(1, 0)), RADIUS, GREEN, -1);
        }
    }


    // used by track class

    void pdaf::paint_validation_gate_points(Mat img, Mat clutter[], int elements) {

        // shows the clutter
        for (int i = 0; i < elements; i++) {
            circle(img, Point(clutter[i].at<float>(0, 0), clutter[i].at<float>(1, 0)), RADIUS, GREEN, -1);
        }

        // shows the validation gate points
        for (int i = 0; i < Mk.elements; i++) {
            circle(img, Point(Mk.Z[i].at<float>(0, 0), Mk.Z[i].at<float>(1, 0)), RADIUS, RED, -1);
        }
    }

    void pdaf::paint_validation_gate_points(Mat img) {

        // shows the validation gate points
        for (int i = 0; i < Mk.elements; i++) {
            circle(img, Point(Mk.Z[i].at<float>(0, 0), Mk.Z[i].at<float>(1, 0)), RADIUS, track_color, -1);
        }
    }

    // obtain the validated measurements from clutter according to epsilon threshold

    void pdaf::print_validation_gate(tvPoints Zk) {
        for (int i = 0; i < Zk.elements; i++) {
            cout << Zk.Z[i] << "(" << Zk.Bi[i] << ")" << ",";
        }
        cout << endl;
    }

    // calulate the Mahalanos distance between the observation and the predicted position according
    // the cov matrix

    float pdaf::Mahalanois(Mat Zk) {
        Mat pZkZk;
        Mat Mahal;

        pZkZk = Zk - pZk;
        Mahal = (pZkZk.t() * Sk.inv() * pZkZk);
        return (sqrt(Mahal.at<float>(0, 0)));

    }

    void pdaf::ValidationGate(bool &create_track, bool &remove_track, int track_idx, bool shared_points) {
        Mat pZkZk;
        Mat V;

        remove_track = false;
        create_track = false;
        Mk.elements = 0;

        for (int i = 0; i < clutter.elements; i++) {
            // If the point was assigned to any other track new iteration
            if ((clutter.assigned[i] != -1) && (!shared_points))
                continue;

            pZkZk = clutter.Z[i] - pZk;
            V = pZkZk.t() * Sk.inv() * pZkZk;

            // cout << "track: " << track_idx << " V: " << V << " pZk: " << pZk << " pZkZk: " << pZkZk << " Sk: " << Sk << endl;

            if (abs(V.at<float>(0, 0)) <= C2) { // we have found measurements inside the Sk of the current track
                Mk.Z[Mk.elements] = clutter.Z[i];
                clutter.assigned[i] = track_idx;
                Mk.Bi[i] = NULL_Bi;

                // cout << "  Zk € VG:" << Mk.Z[Mk.elements] << endl;

                Mk.elements++;

                num_missing = 0;
                num_assigns++;
                create_track = true;
            }
            // else cout << "   no Zk € VG!!" << endl;
        }
        if (Mk.elements == 0) { // measurements not found for actual track
            num_missing++;
            num_assigns = 0;
            if (num_missing > 50) {
                remove_track = true;
            }
        }
    }

    void pdaf::ValidationGate(Mat clutter[], int elements) {
        // var clutter is the clutter elements from tracks class
        // elements is the number of elements from tracks class
        Mat pZkZk;
        Mat V;

        Mk.elements = 0;
        for (int i = 0; i < elements; i++) {
            pZkZk = clutter[i] - pZk;
            V = pZkZk.t() * Sk.inv() * pZkZk;
            if (abs(V.at<float>(0, 0)) <= C2) {
                Mk.Z[Mk.elements] = clutter[i];
                Mk.Bi[i] = NULL_Bi;
                Mk.elements++;
            }
        }
    }

    void pdaf::ValidationGate(tvPoints clutter, tvPoints *validatedZk, Mat pZk, Mat Sk) {
        Mat pZkZk;
        Mat V;

        validatedZk->elements = 0;
        for (int i = 0; i < clutter.elements; i++) {
            pZkZk = clutter.Z[i] - pZk;
            V = pZkZk.t() * Sk.inv() * pZkZk;
            if (abs(V.at<float>(0, 0)) <= C2) {
                validatedZk->Z[validatedZk->elements] = clutter.Z[i];
                validatedZk->Bi[i] = NULL_Bi;
                validatedZk->elements++;
            }
        }
    }

    float pdaf::ValidationGateVolume(Mat Sk) {
        // returns the validation gate volume of Sk
        return (CV_PI * sqrt(abs(determinant(GAMMA * Sk))));
    }

    Mat pdaf::Li(tvPoints Zk, Mat pZk, Mat Sk, float PD, int i) {
        // Zk: measurements within the validation gate
        // pZk: measurement estimate
        // Sk: innovation cov matrix
        // PD: probability of detection
        // i: order number of Bi

        Mat pZkZi;
        float lambda;
        Mat ViZi;
        Mat eViZi;
        Mat NZi;
        Mat ret;

        pZkZi = (Zk.Z[i] - pZk);
        lambda = Zk.elements / ValidationGateVolume(Sk);
        ViZi = pZkZi.t() * Sk.inv() * pZkZi;

        exp(-0.5 * ViZi, eViZi);
        NZi = eViZi / (pow((2 * CV_PI), (2 / 2)) * sqrt(abs(determinant(Sk))));
        ret = NZi * PD / lambda;
        /*
                cout << "Zk=" << Zk.Z[i] << " pZk=" << pZk << endl;
                cout << "pZkZi=" << pZkZi << " lambda=" << lambda << endl;
                cout << "Sk=" << Sk << endl;
                cout << "ViZi=" << ViZi << "eViZi=" << eViZi << endl;
                cout << "NZi=" << NZi << endl;
                cout << "Li: " << ret << endl;
    
                if (abs(ViZi.at<float>(0, 0)) > C2) {
                    print_validation_gate(Zk);
                    cout << "ViZi fuera de VG" << endl;
                }
         */
        if (ret.at<float>(0, 0) > 1.0) ret.at<float>(0, 0) = 1.0;
        return (ret);
    }

    Mat pdaf::sumLi(tvPoints Zk, Mat pZk, Mat Sk, float PD) {
        // return transposed summatory of Li values
        // must be used transposed
        int i;
        Mat sumLi = Mat::zeros(1, 1, CV_32F);

        for (i = 0; i < Zk.elements; i++) {
            sumLi += Li(Zk, pZk, Sk, PD, i);
        }
        return (sumLi);
    }

    Mat pdaf::Bi(tvPoints Zk, Mat pZk, Mat Sk, float PD, int i, bool B0) {
        // Zk: measurements within the validation gate
        // pZk: measurement estimate
        // Sk: innovation cov matrix
        // PD: probability of detection
        // i: order number of Bi

        Mat pZkZi;
        float lambda;
        Mat ViZi;
        Mat eViZi;
        Mat NZi;
        Mat SUMLi;
        Mat retBi;
        Mat L;


        SUMLi = sumLi(Zk, pZk, Sk, PD).t();

        if (B0) { // required to update P(k|k)
            retBi = (1 - PD * PG) / (1 - PD * PG + SUMLi);
        } else { // required to calc. the combined innovation     
            L = Li(Zk, pZk, Sk, PD, i);
            retBi = L / (1 - PD * PG + SUMLi);

            // cout << "L:" << L << endl;
        }

        // cout << "retBi:" << retBi << endl;

        return (retBi);
        // needed to transpote SUM (Li)? no in n=2 because gaussian parametrization are 1x1

    }




    //
    // JPDAF functions
    //

    int factorial(int n) {
        return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
    }

    Mat pdaf::PChiZk(tValMatrix *valMatrix, tvPoints Zk, Mat pZk, Mat Sk, Mat Chi) {
        // Chi: event matrix
        Mat prodFt_j = Mat::ones(1, 1, CV_32F);
        Mat prod_Pt = Mat::ones(1, 1, CV_32F);
        Mat ret = Mat::zeros(1, 1, CV_32F);
        Mat Ftj_factor = Mat(1, 1, CV_32F);

        float ValGateVol = ValidationGateVolume(Sk);
        int false_measurements = num_false_measurements(Chi.cols, Chi.rows, Chi);
        int delta_t;
        double tau_value;

        // cout << "Chi:" << Chi << endl;
        for (int j = 0; j < Chi.rows; j++) { // for each measurement in the event matrix     

            if (tau_value = (double) tau(j, Chi.cols, Chi)) {
                Ftj_factor = Ft_j(valMatrix, Zk, pZk, Sk, Pd, j);
            } else {
                Ftj_factor = Mat::ones(1, 1, CV_32F);
            }
            // tau_value = (double) tau(j, Chi.cols, Chi);
            // pow(Ft_j(valMatrix, Zk, pZk, Sk, Pd, j),tau_value,Ftj_factor);

            prodFt_j = prodFt_j * Ftj_factor;
            //cout << "tau:" << tau_value << " Ftj_factor: " << Ftj_factor << " prodFt_j: " << prodFt_j << endl;
        }

        for (int t = 1; t < Chi.cols; t++) { // for each target
            delta_t = delta(t, Chi.rows, Chi);
            prod_Pt *= pow(Pd, delta_t) * pow((1 - Pd), (1 - delta_t));
        }

        ret = (factorial(false_measurements) / (pow(ValGateVol, false_measurements))) * prodFt_j * prod_Pt;

        // cout << "prodFt_j: " << prodFt_j << "prod_Pt: " << prod_Pt << "ret: " << ret << endl;

        return (ret);
    }

    Mat pdaf::Bjt(tvPoints Zk, Mat pZk, Mat Sk, tValMatrix *valMatrix, int j, int t) {
        Mat sum = Mat::zeros(1, 1, CV_32F);
        valMatrix->c = Mat::zeros(1, 1, CV_32F); // initialization of normalization constant

        // cuidado no sea que el array de matrices de eventos no tenga más que la consolidada (i=0) ¿?
        for (int i = 1; i < valMatrix->events; i++) {
            // cout << "pZk: " << pZk << "Sk: " << Sk << "valMatrix->M[" << i << "]" << valMatrix->M[i] << endl;
            valMatrix->P_Chi[i] = PChiZk(valMatrix, Zk, pZk, Sk, valMatrix->M[i]);
            valMatrix->c += valMatrix->P_Chi[i];
        }

        // calculate the normalization constant
        for (int i = 1; i < valMatrix->events; i++) {
            sum += (valMatrix->P_Chi[i] / valMatrix->c) * valMatrix->M[i].at<float>(j, t);
        }

        return sum;
    }

    float pdaf::B0t(tvPoints Zk, Mat pZk, Mat Sk, tValMatrix *valMatrix, int t) {
        Mat sum = Mat::zeros(1, 1, CV_32F);
        // cout << "B0t" << endl;
        for (int j = 0; j < Zk.elements; j++) {

            if (valMatrix->Bjt_val[j][t].at<float>(0, 0) == NULL_Bi) {
                valMatrix->Bjt_val[j][t] = Bjt(Zk, pZk, Sk, valMatrix, get_measurement_index(valMatrix, Zk.Z[j]), t);
            }
            // cout << "valMatrix->Bjt[" << j << "]" << "[" << t << "]:" << valMatrix->Bjt_val[j][t] << endl;
            sum += valMatrix->Bjt_val[j][t];
        }
        // cout << "sum (B0t): " << sum << endl;
        return (1 - sum.at<float>(0, 0));
    }

    Mat pdaf::get_valPoint(tValMatrix *valMatrix, int j) {
        Mat ret = Mat::zeros(2, 1, CV_32F);

        for (int t = 1; t < valMatrix->M[0].cols; t++) {
            if (valMatrix->M[0].at<float>(j, t) == 1.0) {
                ret = valMatrix->valPoints[j][t];
                break;
            }
        }

        return (ret);
    }

    // returns the index j at the location of M inside valMatrix

    int pdaf::get_measurement_index(tValMatrix *valMatrix, Mat M) {
        int ret = -1;
        int j;

        for (j = 0; j < valMatrix->measurements; j++) {
            if (cv::countNonZero(valMatrix->Mk[j] != M) == 0) {
                ret = j;
                break;
            }
        }

        return (ret);
    }

    int pdaf::num_tracks() {
        return (valMatrix.track_index.num_tracks);
    }

    void pdaf::set_track_idx(int idx, int value) {
        valMatrix.track_index.idx[idx] = value;
    }

    int pdaf::get_track_number(int idx) {
        return (valMatrix.track_index.idx[idx]);
    }

    void pdaf::decrease_num_tracks() {
        valMatrix.track_index.num_tracks--;
    }

    void pdaf::printEventMatrix() {
        string title;

        for (int i = 0; i < valMatrix.events; i++) {
            cout << "M[" << i << "]: " << valMatrix.M[i] << ";" << endl;
            // title = "M[" + static_cast<ostringstream*> (&(ostringstream() << i))->str() + "]";
            // print_matrix(title, Omega.valMatrix.M[i]);
        }
        cout << "valPoints: ";
        for (int j = 0; j < valMatrix.M[0].rows; j++) {
            for (int t = 1; t < valMatrix.M[0].cols; t++) {
                if (valMatrix.valPoints[j][t].rows != 0) {
                    cout << "[" << j << "," << t << "]:" << valMatrix.valPoints[j][t] << ";";
                }
            }
        }
        cout << endl;
    }

    Mat pdaf::Ft_j(tValMatrix *valMatrix, tvPoints Zk, Mat pZk, Mat Sk, float PD, int j) {
        // Zk: measurements within the validation gate
        // pZk: measurement estimate
        // Sk: innovation cov matrix
        // PD: probability of detection
        // j: measurement
        // t: 

        Mat pZkZi;
        Mat ViZi;
        Mat eViZi;

        Mat ret;


        // Zk debe ser el punto asociado a la medida j. Habrá que buscarlo en la
        // matrix valMatrix.valPoints en la coordenada j,t que tenga un uno.
        // pZkZi = (valMatrix.valPoints[j][t] - pZk);

        pZkZi = (get_valPoint(valMatrix, j) - pZk);
        ViZi = pZkZi.t() * Sk.inv() * pZkZi;

        exp(-0.5 * ViZi, eViZi);
        ret = eViZi / (pow((2 * CV_PI), (valMatrix->measurements / 2)) * sqrt(abs(determinant(Sk))));

        // cout << "Zi: " << get_valPoint(valMatrix, j) << " pZk: " << pZk
        //         << "pZkZi: " << pZkZi << " ViZi: " << ViZi << "eViZi: " << eViZi << "ret: " << ret << endl;
        return (ret);

    }


    // adds an event matrix to the valMatrix structure

    void pdaf::add_event_matrix(Mat Chi) {
        if (valMatrix.events == MAX_EVENTS) {
            cout << "OVERFLOW ADDING EVENT MATRIX" << endl;
        } else {
            Chi.copyTo(valMatrix.M[valMatrix.events]);
            valMatrix.events++;
        }
    }


    // Create the base event matrix

    void pdaf::init_event_matrix(int j, int t) {
        valMatrix.M[0] = Mat::zeros(j, t + 1, CV_32F); // This is the validation matrix containing 1 || 0                        
        for (t = 0; t < MAX_TRACKS; t++)
            for (j = 0; j < MAX_VALIDATED_MEASUREMENTS; j++)
                valMatrix.valPoints[j][t] = Mat::zeros(2, 1, CV_32F);

        valMatrix.events = 1;
    }


    // measurement association indicator shows whether measurement j is associated with any target in event Ji

    int pdaf::tau(int j, int T, Mat Chi) {
        // j: number of measurements
        // T: number of tracks 
        // Chi: event matrix
        int sum = 0;

        for (int t = 1; t < T; t++) {
            sum += Chi.at<float>(j, t);
        }
        return (sum);
    }

    // target detection indicator shows whether target t is asociated with any measurement in event Ji

    int pdaf::delta(int t, int m, Mat Chi) {
        // t: number of tracks
        // m: number of measurements
        // Chi: event matrix        
        int sum = 0;


        for (int j = 0; j < m; j++) {
            sum += Chi.at<float>(j, t);
        }
        return (sum);
    }

    // total number of false measurements in event Ji

    int pdaf::num_false_measurements(int t, int m, Mat Chi) {
        // t: number of tracks
        // m: number of measurements
        // Chi: event matrix
        int sum = 0;

        for (int j = 0; j < m; j++) {
            sum += (1 - tau(j, t, Chi));
        }
        return (sum);
    }

    Mat pdaf::innovation_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t) {
        // Zk contains the points inside the validation gate for the current track t

        Mat sum = Mat::zeros(pZk.rows, pZk.cols, CV_32F);

        for (int j = 0; j < Zk.elements; j++) {
            // for each measurement inside the validation gate of the current track

            // cout << "Sk[" << t << "]" << Sk << endl;            

            init_Bjt_values(valMatrix);
            valMatrix->Bjt_val[j][t] = Bjt(Zk, pZk, Sk, valMatrix, get_measurement_index(valMatrix, Zk.Z[j]), t);
            sum += valMatrix->Bjt_val[j][t].at<float>(0, 0) * innovation(Zk.Z[j], pZk);

            // cout << "Zk.Z[" << j << "]" << Zk.Z[j] << "pZk:" << pZk << endl;
            //cout << "Bt(t" << t << " j" << j << "):" << valMatrix->Bjt_val[j][t] << "innovation: " << innovation(Zk.Z[j], pZk) << endl;
        }

        //cout << "sum_innovation(" << t << "):" << sum << endl;

        Yk = sum; // stores the combined innovation

        return (sum);
    }

    Mat pdaf::updateEstimation_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t) {
        // update estimate with all the measurements inside the validation gate
        // pXk: predicted estate
        // Kk: Kalman gain
        // Zk: points inside the validation gate    
        // pZk: predicted measurement
        // Sk: conv matrix of innovation
        // PD: prob. of detection

        Mat innov;


        innov = innovation_JPDAF(Zk, valMatrix, t);
        //cout << "Xk(" << t << ") = pXk + Kk * innov: pXk: " << pXk << "Kk:" << Kk << "innov:" << innov << endl;
        return (pXk + Kk * innov);
    }

    Mat pdaf::updateEstimationErrorCov_JPDAF(tvPoints Zk, tValMatrix *valMatrix, int t) {
        // update error cov matrix of the updated state considering PDAF approach
        // pPk: predicted Cov Matrix of the estimation state
        // Kk: kalman gain
        // Sk: innovation cov matrix
        // Zk: points inside the validation gate
        // pZk: predicted measurement


        float B0;
        Mat Pc; // cov matrix of the stated update with the correct measurement
        Mat Ps; // spread of innovations
        Mat Vj; // innovation of each measurement inside the validation gate
        Mat Vk; // combine innovation
        Mat sum = Mat::zeros(2, 2, CV_32F);

        Mat Pk; // return value

        B0 = B0t(Zk, pZk, Sk, valMatrix, t);
        Pc = pPk - Kk * Sk * Kk.t();

        // cout << "B0t: " << B0 << endl;
        // cout << "Pc: " << Pc << endl;

        Vk = Yk; // from updateEstimation in the previous step


        for (int j = 0; j < Zk.elements; j++) {
            Vj = innovation(Zk.Z[j], pZk);
            if (valMatrix->Bjt_val[j][t].at<float>(0, 0) == NULL_Bi) {
                valMatrix->Bjt_val[j][t] = Bjt(Zk, pZk, Sk, valMatrix, get_measurement_index(valMatrix, Zk.Z[j]), t).at<float>(0, 0);
            }
            // cout << "valMatrix->Bjt[" << j << "]" << "[" << t << "]:" << valMatrix->Bjt_val[j][t] << endl;
            sum += valMatrix->Bjt_val[j][t].at<float>(0, 0) * Vj * Vj.t();
        }
        Ps = Kk * (sum - Vk * Vk.t()) * Kk.t();
        // cout << "Ps: " << Ps << "(sum: " << sum << "Vk: " << Vk << "Kk: " << Kk << ")" << endl;

        Pk = B0 * pPk + (1 - B0) * Pc + Ps;
        // cout << "pPk: " << pPk << endl;
        // cout << "Pk: " << Pk << endl;

        return (Pk);
    }

    void pdaf::reset_track_index() {
        valMatrix.track_index.num_tracks = 0;
        for (int t = 0; t < 10; t++)
            valMatrix.track_index.idx[t] = -1;
    }

    void pdaf::add_track_index(int t) {
        valMatrix.track_index.idx[valMatrix.track_index.num_tracks] = t;
        valMatrix.track_index.num_tracks++;
    }

    // return the idx of track t inside the track_index structure or -1 if not found    

    int pdaf::get_track_index(int t) {
        int idx = -1;

        for (idx = 0; idx < valMatrix.track_index.num_tracks; idx++) {
            if (valMatrix.track_index.idx[idx] == t)
                break;
        }
        return (idx == valMatrix.track_index.num_tracks ? -1 : idx);
    }

    void pdaf::reset_valMatrix() {
        valMatrix.measurements = 0;
        valMatrix.events = 0;
    }

    int pdaf::locate_validated_point(Mat p) {

        int ret = -1;

        for (int i = 0; i < valMatrix.measurements; i++) {
            if (p.at<float>(0, 0) == valMatrix.Mk[i].at<float>(0, 0) &&
                    p.at<float>(1, 0) == valMatrix.Mk[i].at<float>(1, 0)) {
                ret = i;
                break;
            }
        }

        return (ret);
    }

    // return the number of unique validated measurements considering the Validation Gate of all the active tracks

    int pdaf::num_validated_measurements() {
        return (valMatrix.measurements);
    }

    void pdaf::add_validated_point(Mat p) {
        // inserts unique validated points into the valMatrix structure
        if (locate_validated_point(p) == -1) {
            valMatrix.Mk[valMatrix.measurements] = (Mat_<float>(2, 1) << p.at<float>(0, 0), p.at<float>(1, 0));
            valMatrix.measurements++;
        }
    }

    void pdaf::printEventMatrix(tValMatrix *valMatrix) {
        string title;
        cout << "*****" << endl;
        for (int i = 0; i < valMatrix->events; i++) {
            // title = "M[" + static_cast<ostringstream*> (&(ostringstream() << i))->str() + "]";
            // print_matrix(title, Omega.valMatrix.M[i]);
        }
        cout << "valPoints" << endl;
        for (int j = 0; j < valMatrix->M[0].rows; j++) {
            for (int t = 1; t < valMatrix->M[0].cols; t++) {
                if (valMatrix->valPoints[j][t].rows != 0) {
                    cout << "[" << j << "," << t << "]:" << valMatrix->valPoints[j][t] << ";";
                }
            }
        }
        cout << endl;
    }

    void pdaf::set_valMatrix_value(int M_idx, int j, int t, float value) {
        valMatrix.M[M_idx].at<float>(j, t) = value;
    }

    int pdaf::get_num_cols_event_matrix() {
        return (valMatrix.M[0].cols);
    }

    float pdaf::get_valMatrix_element(int M_idx, int j, int t) {
        return (valMatrix.M[M_idx].at<float>(j, t));
    }

    void pdaf::insert_valPoint(int j, int track_idx, Mat M) {
        M.copyTo(valMatrix.valPoints[j][track_idx]);
    }

    void pdaf::set_stop_button() {
        movButton = STOP;
        step = Point(0, 0);
    }

    void pdaf::set_up_button() {
        movButton = UP;
        step = Point(0, -1);
    }

    void pdaf::set_down_button() {
        movButton = DOWN;
        step = Point(0, 1);
    }

    void pdaf::set_left_button() {
        movButton = LEFT;
        step = Point(-1, 0);
    }

    void pdaf::set_right_button() {
        movButton = RIGHT;
        step = Point(1, 0);
    }

    void pdaf::set_upright_button() {
        movButton = UPRIGHT;
        step = Point(1, -1);
    }

    void pdaf::set_downright_button() {
        movButton = DOWNRIGHT;
        step = Point(1, 1);
    }

    void pdaf::set_upleft_button() {
        movButton = UPLEFT;
        step = Point(-1, -1);
    }

    void pdaf::set_downleft_button() {
        movButton = DOWNLEFT;
        step = Point(-1, 1);
    }

    void pdaf::set_home_button() {
        movButton = DOWNLEFT;
        c = Point(MAXX / 2, MAXY / 2);
    }

    bool pdaf::stop_button() {
        return (movButton == STOP ? 1 : 0);
    }

    bool pdaf::up_button() {
        return (movButton == UP ? 1 : 0);
    }

    bool pdaf::down_button() {
        return (movButton == DOWN ? 1 : 0);
    }

    bool pdaf::left_button() {
        return (movButton == LEFT ? 1 : 0);
    }

    bool pdaf::right_button() {
        return (movButton == RIGHT ? 1 : 0);
    }

    bool pdaf::upleft_button() {
        return (movButton == UPLEFT ? 1 : 0);
    }

    bool pdaf::downleft_button() {
        return (movButton == DOWNLEFT ? 1 : 0);
    }

    bool pdaf::upright_button() {
        return (movButton == UPRIGHT ? 1 : 0);
    }

    bool pdaf::downright_button() {
        return (movButton == DOWNRIGHT ? 1 : 0);
    }
}

