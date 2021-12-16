#include <opencv2/opencv.hpp>
#include <iostream>

#include "dilatacion.hpp"

using namespace cv;



Mat dilatacion(Mat original_image) {
	Mat dilate_image,erode_image,difference_image, difference_image_2;

	int erosion_size = 10;
	Mat element = getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	dilate(original_image, dilate_image,element);
	erode(dilate_image, erode_image, element);
        
        // erode(original_image, erode_image,element);
	// dilate(erode_image, dilate_image, element);
        
/*
 
 erode(original_image, erode_image,element);
	dilate(erode_image, dilate_image, element);
 * ts = 0.04
 * Pd = 0.25
 */
        
        
        return dilate_image;

/*
	absdiff(original_image,erode_image ,difference_image);
	absdiff(original_image, dilate_image, difference_image_2);

	namedWindow("ORIGINAL IMAGE", WINDOW_AUTOSIZE);
	namedWindow("DILATE IMAGE", WINDOW_AUTOSIZE);
	namedWindow("ERODE IMAGE AFTER DILATE", WINDOW_AUTOSIZE);
	namedWindow("DIFFERENCE BETWEEN TWO IMAGES", WINDOW_AUTOSIZE);
	namedWindow("DIFFERENCE BETWEEN TWO IMAGES 2", WINDOW_AUTOSIZE);

	imshow("ORIGINAL IMAGE", original_image);
	imshow("DILATE IMAGE", dilate_image);
	imshow("ERODE IMAGE AFTER DILATE", erode_image);
	imshow("DIFFERENCE BETWEEN TWO IMAGES",difference_image);
	imshow("DIFFERENCE BETWEEN TWO IMAGES 2", difference_image_2);


	waitKey(0);
 

	original_image.release();
	dilate_image.release();
	erode_image.release();
	difference_image.release();
	difference_image_2.release();

	destroyAllWindows();
*/
}