/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   num_blobs.hpp
 * Author: Luis Menendez (luis.menendez@gmail.com)
 *
 * Created on November 29, 2019, 7:04 PM
 */


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


#ifndef NUM_BLOBS_HPP
#define NUM_BLOBS_HPP

vector<Point2i> blobs(Mat frame);


#endif /* NUM_BLOBS_HPP */

