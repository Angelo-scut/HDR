#pragma once
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  This is Zishun Wang's private programming.If you want to use this,
//  you should ask me for permission.
//  OpenCV version:3.2
//  Compilier version C++11
//  Library including:
//         1、HDR merge and tonemapping                 
//         2、Auto exposure       
//
// Copyright (C) 2020-, South China University of Technology, all rights reserved.
// Third party copyrights are property of their respective owners.
//
//M*/
#ifndef ANGELOINSCUT_HPP
#define ANGELOINSCUT_HPP
#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <cstdlib>
#include <string>
#include <cmath>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
namespace zs {
	using namespace cv;
#define MV_OK                   0x00000000  ///< 成功，无错误 | en:Successed, no error
#define MV_E_UNKNOW             0x800000FF  ///< 未知的错误 | en:Unknown error
#define MV_NEEDINIT             0x80000FFF  ///< 需要初始化 | en:need init
#define MV_NOTNEEDHDR           0x80001FFF  ///< 不需要HDR | en:no need HDR
#define MV_UNDEREXPO            0x80002FFF  ///< 欠曝光 | en:under exposure
#define COMPRESSHIGH            3.0f  ///< 压缩高亮度区域 | en:compress high pixel
#define COMPRESSLOW             0.4f  ///< 压缩低亮度区域 | en:compress low pixel
class zsHDR
{
public:
	zsHDR():name("MergeZishun"),
		pixel(0)
		//overweight(overWeights())
	{
	}
	//~zsHDR();
	void merge(InputArrayOfArrays src, OutputArray dst, InputArray _times, float compress);
	void mergewithcuda(InputArrayOfArrays src, OutputArray dst, InputArray _times, float compress);
	int AutoExposure(Mat src, uint exposure_time, uint& next_time, uchar &order);
	void Hist(Mat src, Mat& dst);
	void Arbitaryhist(Mat src, Mat&dst, int binnumber);
	void Histmapping(Mat src, Mat dst);
	Mat maskWeight(uchar maskpixel);
	void Histshow(Mat src, Mat& dst);
protected:
	String name;
	uchar pixel;
	//Mat overweight;
};


void checkImageDimensions(const std::vector<Mat>& images);
Mat mytringleWeights(uchar order,uchar low=0,uchar high=255);
Mat tringleWeights(uchar order);
Mat lowWeights();
Mat midWeights();
Mat highWeights();
Mat overWeights();
Mat selfWeights(uchar p);
Mat CurveResponse(float ratio);
}

#endif