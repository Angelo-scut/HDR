#include "Angeloinscut.h"

namespace zs {
	using namespace std;
void checkImageDimensions(const std::vector<Mat>& images)
{
	CV_Assert(!images.empty());
	int width = images[0].cols;
	int height = images[0].rows;
	int type = images[0].type();

	for (size_t i = 0; i < images.size(); i++) {
		CV_Assert(images[i].cols == width && images[i].rows == height);
		CV_Assert(images[i].type() == type);
	}
}

Mat mytringleWeights(uchar order, uchar low, uchar high)
{
	Mat w(LDR_SIZE, 1, CV_32F);
	int half = LDR_SIZE / 2;
	uchar range = 64;
	float eqs = 0.01f;
	switch (order)
	{
	case 0:
		if (low > range) {
			for (int i = 0; i < low- range; i++) {
				w.at<float>(i) = eqs;
			}
			for (int i = low - range; i < low+ range; i++) {
				w.at<float>(i) = float(i- low + range)/ (2*range) + eqs;
			}
			for (int i = low + range; i < LDR_SIZE; i++) {
				w.at<float>(i) = 1 + eqs;
			}
		}
		else{
			for (int i = 0; i < low + range; i++) {
				w.at<float>(i) = float(i) / (low+ range) + eqs;
			}
			for (int i = low + range; i < LDR_SIZE; i++) {
				w.at<float>(i) = 1 + eqs;
			}
		}
		break;
	case 1:
		if (high < LDR_SIZE- range) {
			if (low>range)
			{
				for (int i = 0; i < low - range; i++) {
					w.at<float>(i) = eqs;
				}
				for (int i = low - range; i < low + range; i++) {
					w.at<float>(i) = float(i - low + range) / (2* range) + eqs;
				}
				for (int i = low + range; i < high - range; i++) {
					w.at<float>(i) = 1 + eqs;
				}
				for (int i = high - range; i < high + range; i++) {
					w.at<float>(i) = float(high - range - i) / (2* range) + 1+ eqs;
				}
				for (int i = high + range; i < LDR_SIZE; i++) {
					w.at<float>(i) = eqs;
				}
			}
			else
			{
				for (int i = 0; i < low + range; i++) {
					w.at<float>(i) = float(i) / (low + range) + eqs;
				}
				for (int i = low + range; i < high - range; i++) {
					w.at<float>(i) = 1+ eqs;
				}
				for (int i = high - range; i < high + range; i++) {
					w.at<float>(i) = float(high - range - i) / (2* range) + 1+ eqs;
				}
				for (int i = high + range; i < LDR_SIZE; i++) {
					w.at<float>(i) = eqs;
				}
			}
		}
		else {
			if (low>range)
			{
				for (int i = 0; i < low - range; i++) {
					w.at<float>(i) = eqs;
				}
				for (int i = low - range; i < low + range; i++) {
					w.at<float>(i) = float(i - low + range) / (2* range) + eqs;
				}
				for (int i = low + range; i < high - range; i++) {
					w.at<float>(i) = 1+ eqs;
				}
				for (int i = high - range; i < LDR_SIZE; i++) {
					w.at<float>(i) = float(high - range - i) / (LDR_SIZE - high + range) + 1+ eqs;
				}
			}
			else
			{
				for (int i = 0; i < low + range; i++) {
					w.at<float>(i) = float(i) / (low + range) + eqs;
				}
				for (int i = low + range; i < high - range; i++) {
					w.at<float>(i) = 1+ eqs;
				}
				for (int i = high - range; i < LDR_SIZE; i++) {
					w.at<float>(i) = float(high - range - i) / (LDR_SIZE - high + range) + 1+ eqs;
				}
			}
		}
		break;
	case 2:
		if (high < LDR_SIZE- range) {
			for (int i = 0; i < high - range; i++) {
				w.at<float>(i) = 1+ eqs;
			}
			for (int i = high - range; i < high + range; i++) {
				w.at<float>(i) = float(high - range - i) / (2* range) + 1+ eqs;
			}
			for (int i = high + range; i < LDR_SIZE; i++) {
				w.at<float>(i) = eqs;
			}
		}
		else {
			for (int i = 0; i < high - range; i++) {
				w.at<float>(i) = 1+ eqs;
			}
			for (int i = high - range; i < LDR_SIZE; i++) {
				w.at<float>(i) = float(high - range - i) / (LDR_SIZE-high+ range) + 1 + eqs;
			}
		}
		break;
	default:
		break;
	}
	return w;
}

Mat tringleWeights(uchar order)
{
	Mat w(LDR_SIZE, 1, CV_32F);
	int half = LDR_SIZE / 2;
	switch (order)
	{
	case 0:
		for (int i = 0; i < LDR_SIZE; i++) {
			w.at<float>(i) = i < half ? float(i) : LDR_SIZE - i;
		}
		break;
	case 1:
		for (int i = 0; i < LDR_SIZE; i++) {
			w.at<float>(i) = i < half ? float(i) : LDR_SIZE - i-1;
		}
		break;
	case 2:
		for (int i = 0; i < LDR_SIZE; i++) {
			w.at<float>(i) = i < half ? float(i) : LDR_SIZE - i-1;
		}
		break;
	default:
		break;
	}
	return w;
}

Mat lowWeights()
{
	Mat w(LDR_SIZE, 1, CV_32FC1);
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<float>(i) = i < 30 ? 1.0f : 0.0f;
	}
	return w;
}

Mat midWeights()
{
	Mat w(LDR_SIZE, 1, CV_32FC1);
	for (int i = 0; i < 128; i++) {
		w.at<float>(i) = i < 30 ? 0.0f : 1.0f;
	}
	for (int i = 128; i < LDR_SIZE; i++) {
		w.at<float>(i) = i > 240 ? 0.0f : 1.0f;
	}
	return w;
}

Mat highWeights()
{
	Mat w(LDR_SIZE, 1, CV_32FC1);
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<float>(i) = i > 240 ? 1.0f : 0.0f;
	}
	return w;
}

Mat overWeights()
{
	Mat w(LDR_SIZE, 1, CV_8UC1);
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<uchar>(i) = i == 255 ? 0 : 1;
	}
	return w;
}

Mat selfWeights(uchar p)
{
	Mat w(LDR_SIZE, 1, CV_32FC1);
	float temp = float(p);
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<float>(i) = i == temp ? 1.0f : 0.0f;
	}
	return w;
}

Mat CurveResponse(float ratio)
{
	Mat w(LDR_SIZE, 1, CV_32F);
	float a = 1 / ratio;
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<float>(i) = a * i / 255.0;
	}
	return w;
}


void zsHDR::merge(InputArrayOfArrays src, OutputArray dst, InputArray _times,float compress)
{
	std::vector<Mat> images;
	src.getMatVector(images);
	Mat times = _times.getMat();

	CV_Assert(images.size() == times.total());
	CV_Assert(images.size() == 3);
	checkImageDimensions(images);
	CV_Assert(images[0].depth() == CV_8U);
	CV_Assert(images[0].channels() == 1);//目前还只是能够融合3张通道为1的灰度图像

	int channels = images[0].channels();
	Size size = images[0].size();
	int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

	dst.create(images[0].size(), CV_32FCC);
	Mat result = dst.getMat();//将传入参数转换为Mat结构
	result = Mat::zeros(size, CV_32FCC);

	vector<Mat> mask(images.size(), Mat());
	vector<Mat> edgemask(2, Mat());
	Mat finalmask,opmask;
	//Mat weightmask = Mat::zeros(size, CV_32FCC);
	mask[0] = Mat::zeros(size, CV_32FCC);
	Mat overweight = overWeights();
	vector<Mat> weights;
	//for (size_t i = 0; i < images.size(); i++)
	//{
	//	weights.push_back(tringleWeights(i));
	//}
	Mat weight_sum = Mat::zeros(size, CV_32FCC);
	vector<Mat> Response(images.size(), Mat());
	//vector<Mat> ResponseCurve(images.size(), Mat());
	vector<float> Ratio(images.size(),0.0);
	//vector<float> Ratio_sum(images.size(), 0.0);
	Ratio[0] = 1.0;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));//定义一个膨胀核
	for (size_t i = 1; i < images.size(); i++)
	{
		LUT(images[i], overweight, edgemask[i-1]);
		erode(edgemask[i-1], mask[i], element);
		edgemask[i - 1] = edgemask[i - 1] - mask[i];
		//dilate(edgemask[i - 1], edgemask[i - 1], element);
		//Ratio[i] = times.at<float>(i) / times.at<float>(0);
	}
	float myhigh = 0.0f;
	for (size_t i = 0; i < edgemask.size(); i++)
	{
		Mat temp;// , temphist;
		float avg1, avg2;
		//Point maxp;
		int n1, n2;
		edgemask[i].convertTo(temp, images[0].type());
		temp = images[i].mul(temp);
		//Hist(temp, temphist);
		//temphist.rowRange(1,255).copyTo(temphist);
		//minMaxLoc(temphist, NULL, NULL, NULL, &maxp);
		double s = cv::sum(temp)[0];
		n1 = countNonZero(temp);
		avg1 = s / n1;
		//n1 = maxp.y;
		edgemask[i].convertTo(temp, images[0].type());
		temp = images[i+1].mul(temp);
		//Hist(temp, temphist);
		//temphist.rowRange(1, 255).copyTo(temphist);
		//minMaxLoc(temphist, NULL, NULL, NULL, &maxp);
		s = cv::sum(temp)[0];
		n2 = countNonZero(temp);
		avg2 = s / n2;
		Ratio[i + 1] = ((avg2+avg1)*Ratio[i] / avg1);
		//dilate(edgemask[i], edgemask[i], element);
		//n2 = maxp.y;
		//Ratio[i+1] = (n2*Ratio[i] / n1) + 1;
		//cout << Ratio[i + 1] <<' '<< avg2<< ' '<< avg1 <<endl;
		switch (i)
		{
		case 0:
			weights.push_back(mytringleWeights(i,uchar(avg1)));
			myhigh = avg2;
			break;
		case 1:
			weights.push_back(mytringleWeights(i, uchar(avg1), uchar(myhigh)));
			weights.push_back(mytringleWeights(i+1, 0, uchar(avg2)));
			break;
		default:
			break;
		}
	}
	//finalmask = mask[1] + mask[2];
	//vector<Mat> edgemask(2, Mat());
	//element = getStructuringElement(MORPH_RECT, Size(3, 3));
	/*for (int i = 1; i < images.size(); i++)
	{
		Mat dxgraident,temphist;
		Point maxp;
		int n1, n2;
		float avg1, avg2;
		erode(mask[i], dxgraident, element);
		dxgraident = mask[i] - dxgraident;
		dxgraident.copyTo(edgemask[i - 1]);
		//dxgraident.convertTo(dxgraident, images[i].type());
		//dxgraident = images[i].mul(dxgraident);
		//Hist(dxgraident, temphist);
		//temphist.rowRange(1,256).copyTo(temphist);
		//minMaxLoc(temphist, NULL, NULL, NULL, &maxp);
		//n1 = maxp.y;
		//cout << maxp << endl;
		//dxgraident.copyTo(GauMask[i - 1]);
		//double s = cv::sum(dxgraident)[0];
		//n1 = countNonZero(dxgraident);
		//avg1 = s / n1;
		//cout << avg1 << endl;
		dilate(mask[i], dxgraident, element);
		dxgraident = dxgraident - mask[i];
		edgemask[i - 1] += dxgraident;
		//dxgraident.convertTo(dxgraident, images[i].type());
		//dxgraident = images[i-1].mul(dxgraident);
		//Hist(dxgraident, temphist);
		//temphist.rowRange(1, 256).copyTo(temphist);
		//minMaxLoc(temphist, NULL, NULL, NULL, &maxp);
		//n2 = maxp.y;
		//Ratio[i] = (n1*Ratio[i - 1] / n2)+1;
		//GauMask[i - 1] = GauMask[i - 1] + dxgraident;	
		//s = cv::sum(dxgraident)[0];
		//n2 = countNonZero(dxgraident);
		//avg2 = s / n2;
		//Ratio[i] = (avg1*Ratio[i - 1] / avg2)+1;
		//double s = 0.0;
		//int n = 0;;
		//float rat = 0.0f;
		//for (uchar j = 1; j < 8; j++)
		//{
		//	Mat tempweight = selfWeights(j * 3);
		//	Mat temp;
		//	LUT(images[i-1], tempweight, temp);
		//	temp.convertTo(temp, images[i].type());
		//	temp = images[i].mul(temp);
		//	s = cv::sum(temp)[0];
		//	n = countNonZero(temp);
		//	rat += (s / n - j * 5) / (j * 5);
		//}
		//Ratio[i] = Ratio[i-1] * rat / 7;
		//cout << Ratio[i] << endl;
	}*/
	//Mat emask,opemask;
	//edgemask[0].copyTo(emask);
	//emask += edgemask[1];
	//edgemask[0].release();
	//edgemask[1].release();
	for (size_t i = 0; i < images.size(); i++)
	{
		Mat ResponseCurve = CurveResponse(Ratio[i]);
		LUT(images[i], ResponseCurve, Response[i]);
	}
	
	Mat w = Mat::zeros(size, CV_32FCC);
	for (size_t i = 0; i < images.size(); i++)
	{
		LUT(images[i], weights[i], w);
		Response[i] = Response[i].mul(w);
		weight_sum += w;
		result += Response[i];
	}
	weight_sum = 1 / weight_sum;
	result = result.mul(weight_sum);
	//for (int i = images.size()-1; i > -1; i--)
	//{
	//	Mat temp = Mat::ones(size, CV_32FCC);
	//	Mat w = Mat::zeros(size, CV_32FCC);
	//	switch (i)
	//	{
	//	case 0:
	//		for (size_t j = i + 1; j < images.size(); j++)
	//			mask[i] += mask[j];
	//		mask[i] = mask[i] ^ temp;		
	//		//opmask = finalmask ^ temp;
	//		Response[i] = Response[i].mul(mask[i]);
	//		break;
	//	case 1:
	//		temp = temp ^ mask[i + 1];
	//		mask[i] = mask[i] & temp;
	//		LUT(images[i], weights, w);
	//		temp = Response[i].mul(w);
	//		weight_sum += w;
	//		LUT(images[i - 1], weights, w);
	//		Response[i] = Response[i - 1].mul(w);
	//		weight_sum += w;
	//		weight_sum = 1 / weight_sum;
	//		Response[i] += temp;
	//		Response[i] = Response[i].mul(weight_sum);
	//		Response[i] = Response[i].mul(mask[i]);
	//		break;
	//	case 2:
	//		LUT(images[i], weights, w);
	//		temp = Response[i].mul(w);
	//		weight_sum += w;
	//		LUT(images[i-1], weights, w);
	//		Response[i] = Response[i-1].mul(w);
	//		weight_sum += w;
	//		weight_sum = 1 / weight_sum;
	//		Response[i] += temp;
	//		Response[i] = Response[i].mul(weight_sum);
	//		Response[i] = Response[i].mul(mask[i]);
	//		break;
	//	default:
	//		break;
	//	}
	//	result += Response[i];
	//}
	//for (size_t i = 0; i < edgemask.size(); i++)
	//{
	//	Mat temp;
	//	temp = Response[i].mul(edgemask[i]);
	//	temp += Response[i+1].mul(edgemask[i]);
	//	temp = temp.mul(0.5);
	//	weightmask += temp;
	//}
	//result = result.mul(opmask);
	//result += weightmask;
	compress = 1.0f / compress;
	pow(result, compress, result);
	double minval, maxval;
	minMaxLoc(result,&minval,&maxval);
	minval = float(minval);
	float interval = 1.0f / (maxval - minval);
	result = result - minval;
	result = result.mul(interval);
	minMaxLoc(result, &minval, &maxval);
	//Mat hist;
	//Arbitaryhist(result, hist, 256);
	//cout << hist << endl;
	//float len = 1.0f / (result.rows*result.cols);
	//hist = hist.mul(len);
	//size = hist.size();
	//Mat CDF = Mat::zeros(size,CV_32FC1);
	//CDF.at<float>(0) = hist.at<float>(0);
	//result = result.mul(256);
	//for (size_t i = 1; i < hist.rows; i++)
	//{
	//	CDF.at<float>(i) = CDF.at<float>(i-1) + hist.at<float>(i);
	//}
	//CDF = CDF.mul(CDF.rows);
	//for (size_t i = 0; i < result.rows; i++)
	//{
	//	for (size_t j = 0; j < result.cols; j++)
	//	{
	//		result.at<float>(i, j) = CDF.at<float>(result.at<float>(i, j));
	//	}
	//}
	//result = result.mul(1.0f / 256);
	//cout << CDF << endl;

	//result = 255 * result;
	//result.convertTo(result, images[0].type());
	//Mat temp;
	//blur(result, temp,Size(27,27));
	//temp = temp.mul(emask);
	//result = result.mul(opemask);
	//element = getStructuringElement(MORPH_RECT, Size(4, 4));
	//dilate(result, result, element);
	//result += temp;
	//GaussianBlur(result, temp, Size(7, 7), 10000000);
	//temp.release();
	//Histmapping(result, result);
	return;
}

void zsHDR::mergewithcuda(InputArrayOfArrays src, OutputArray dst, InputArray _times, float compress)
{
	clock_t start, end;
	std::vector<Mat> images;
	
	src.getMatVector(images);
	Mat times = _times.getMat();

	CV_Assert(images.size() == times.total());
	CV_Assert(images.size() == 3);
	checkImageDimensions(images);
	CV_Assert(images[0].depth() == CV_8U);
	CV_Assert(images[0].channels() == 1);//目前还只是能够融合3张通道为1的灰度图像

	int channels = images[0].channels();
	Size size = images[0].size();
	int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

	dst.create(images[0].size(), CV_32FCC);
	Mat result = dst.getMat();//将传入参数转换为Mat结构
	result = Mat::zeros(size, CV_32FCC);

	cuda::GpuMat gputemp;
	vector<cuda::GpuMat> gpuedgemask(2, cuda::GpuMat());
	cuda::GpuMat gpuweight_sum;
	gpuweight_sum.upload(Mat::zeros(size, CV_32FCC));
	cuda::GpuMat gpuresult;
	gpuresult.upload(Mat::zeros(size, CV_32FCC));
	vector<cuda::GpuMat> gpuweights;
	vector<cuda::GpuMat> gpuResponse(images.size(), cuda::GpuMat());
	cuda::GpuMat gpukernel;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));//定义一个膨胀核
	Ptr<cuda::Filter> gpuerode = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, element);
	vector<cuda::GpuMat> gpuim(images.size(), cuda::GpuMat());
	for (uchar i = 0; i < images.size(); i++)
	{
		gpuim[i].upload(images[i]);
	}
	Mat mytemp;

	Mat overweight = overWeights();
	vector<Mat> weights;
	vector<float> Ratio(images.size(), 0.0);
	Ratio[0] = 1.0;
	Ptr<cuda::LookUpTable> overlookup = cuda::createLookUpTable(overWeights());

	start = clock();
	for (size_t i = 1; i < images.size(); i++)
	{
		overlookup->transform(gpuim[i], gpuedgemask[i - 1]);
		gpuerode->apply(gpuedgemask[i-1], gputemp);
		cuda::subtract(gpuedgemask[i-1], gputemp, gpuedgemask[i-1]);
	}
	float myhigh = 0.0f;
	for (size_t i = 0; i < gpuedgemask.size(); i++)
	{
		Mat temp;
		float avg1, avg2;
		int n1, n2;
		cuda::multiply(gpuim[i], gpuedgemask[i], gputemp);
		Scalar gpus = cuda::sum(gputemp);
		n1 = cuda::countNonZero(gputemp);
		double s=gpus.val[0];
		avg1 = s / n1;
		cuda::multiply(gpuim[i+1], gpuedgemask[i], gputemp);
		gpus = cuda::sum(gputemp);
		n2 = cuda::countNonZero(gputemp);
		s = gpus.val[0];
		avg2 = s / n2;
		Ratio[i + 1] = ((avg2 + avg1)*Ratio[i] / avg1);
		switch (i)
		{
		case 0:
			weights.push_back(mytringleWeights(i, uchar(avg1)));
			myhigh = avg2;
			break;
		case 1:
			weights.push_back(mytringleWeights(i, uchar(avg1), uchar(myhigh)));
			weights.push_back(mytringleWeights(i + 1, 0, uchar(avg2)));
			break;
		default:
			break;
		}
	}
	for (size_t i = 0; i < images.size(); i++)
	{
		gpuim[i].convertTo(gputemp, CV_32FC1);
		float temp = 1.0f / (255.0f*Ratio[i]);
		cuda::multiply(gputemp, temp, gpuResponse[i]);
	}
	for (size_t i = 0; i < images.size(); i++)
	{
		LUT(images[i], weights[i], mytemp);
		gputemp.upload(mytemp);
		cuda::multiply(gpuResponse[i], gputemp, gpuResponse[i]);
		cuda::add(gpuweight_sum, gputemp, gpuweight_sum);
		cuda::add(gpuresult, gpuResponse[i], gpuresult);
	}
	compress = 1.0f / compress;
	cuda::divide(1.0f, gpuweight_sum, gpuweight_sum);
	cuda::multiply(gpuresult, gpuweight_sum, gpuresult);
	cuda::pow(gpuresult, compress, gpuresult);

	double minval, maxval;
	cuda::minMax(gpuresult, &minval, &maxval);
	float interval = 1.0f / (maxval - minval);
	cuda::subtract(gpuresult, minval, gpuresult);
	cuda::multiply(gpuresult, interval, gpuresult);
	gpuresult.download(result);
	end = clock();
	cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
	return;
}

int zsHDR::AutoExposure(Mat src, uint exposure_time, uint & next_time, uchar &order)
{
	
	int nRet = MV_OK;
	uint temp_time = 0, temp_sum = 0;
	uchar temp_pixel = 0;
	Mat hist;
	Hist(src, hist);
	float* temp_hist = (float*)hist.data;
	uint length = src.cols * src.rows;
	if (order == 0)//只是要判断试一下是否正确曝光而已，不能直接返回，还要计算下一个曝光时间呢
	{
		if (pixel == 0)//如果等于零表示刚初始化过，刚初始化过没必要再判断是否需要初始化，除非严重欠曝
			pixel = temp_hist[255];
		for (uint i = 255; i > 0; i--)//统计前0.1%的像素位置
		{
			temp_sum += temp_hist[i];
			if (temp_sum > 0.001*length)
			{
				temp_pixel = i;
				break;
			}
		}
		if ((temp_pixel == 255 && abs(temp_sum - pixel) > 0.001*length) || temp_pixel < 100)//过曝太多了或者欠曝太多了,则重新initial一下..虽然解决了能够在非HDR环境不使用HDR曝光了，但是好像
		{
			nRet = MV_NEEDINIT;//为什么没办法自动转换成HDR模式了？
			pixel = 0;
			return nRet;
		}
		else if (temp_pixel < 200)
		{
			nRet = MV_UNDEREXPO;//如果第一张欠曝光了，则重新调整曝光序列
			pixel = 0;
			next_time = exposure_time * (240 / temp_pixel);//(response_curve.at<float>(249) / response_curve.at<float>(temp_pixel))*exposure_time;//因为response是已经含e的指数部分了
			if (next_time > 50000)
			{
				next_time = 50000;
				nRet = MV_NOTNEEDHDR;
			}
			return nRet;
		}
		else
		{
			temp_sum = 0;
			pixel = temp_hist[255];
			for (uint i = 0; i < 20; i++)
				temp_sum += temp_hist[i];
			if (temp_sum < 0.08*length)//表示欠曝光像素值较少，无需使用HDR
			{
				nRet = MV_NOTNEEDHDR;
				return nRet;
			}
		}
	}
	//uchar Threshold = 20;//好像30太高了点，导致低亮度值的地方依旧没能获得比较良好的曝光,可以考虑10或者20，得慢慢调一下,调参的时候发现，最后一张30的时候会比较好
	//if (order == 1)//因为order是从0开始的，所以0代表着第一帧，1代表第二帧，第三帧已经不再需要调光了
	//	Threshold = 30;
	//temp_pixel = Threshold;//但是如果手动调超参数好像场景适应性不强，有什么办法能够
	temp_sum = 0;
	for (uint i = 249; i >0; i--)//过曝的不再需要
	{
		temp_sum += temp_hist[i];
		if (temp_sum > 0.3*length)
		{
			//if (i > 2 * Threshold)
			//	temp_pixel = 1.2*Threshold;
			//else if (i>Threshold)
			temp_pixel = i;
			break;
		}
	}
	next_time = exposure_time * (floor(240 / temp_pixel)>8? 8: floor(240 / temp_pixel));//8是个超参数，只需要将动态范围扩大64倍即可达到10的5次方
	if (next_time > 50000)
	{
		next_time = 50000;
	}
	return nRet;
}

void zsHDR::Hist(Mat src, Mat& dst)
{
	int histBinNum = 256;
	float range[] = { 0,256 };
	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), dst, 1, &histBinNum, &histRange);//计算直方图
	return;
}

void zsHDR::Arbitaryhist(Mat src, Mat & dst, int binnumber)
{
	int histBinNum = binnumber;
	float range[] = { 0.0f,1.0f };
	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), dst, 1, &histBinNum, &histRange);//计算直方图
	return;
}

void zsHDR::Histmapping(Mat src, Mat dst)
{
	Mat image = 255 * src;
	image.convertTo(image, CV_8UC1);
	Mat hist;
	Hist(image, hist);
	float* temp_hist = (float*)hist.data;
	uint length = src.cols*src.rows;
	uint temp_sum = 0;
	uint i = 0;
	for (; temp_sum < length / 6; i++) {
		temp_sum += temp_hist[i];
	}
	if (i<3)
	{
		i = 3;
	}
	Mat maskH;
	Mat maskL = Mat::ones(image.size(),CV_32FC1);
	LUT(image, maskWeight(i), maskH);
	maskL = maskH ^ maskL;
	Mat tempmask;
	maskH.copyTo(tempmask);
	maskH = maskH.mul(src);
	maskL = maskL.mul(src);
	float Zmin = float(i) / LDR_SIZE;
	float a = -(0.25f / (Zmin *Zmin));
	float b = 0.5f /  Zmin;
	Mat temp;
	pow(maskL, 2, temp);
	maskL = a * temp + b * maskL;
	float Z1 = (1 - Zmin)*(1 - Zmin);
	a = 0.75f / Z1;
	b = -(1.5f * Zmin / Z1);
	float c = 1 - (b + a);
	pow(maskH, 2, temp);
	maskH = a * temp + b * maskH + c * tempmask;
	imshow("tempmask", c * tempmask);
	//pow(maskH, 1.0f / 3.0, maskH);
	//pow(maskL, 1.0f / 3.0, maskL);
	//imshow("maskH", maskH);
	//imshow("maskL", maskL);
	//tempmask = maskH + maskL;
	//imshow("tempmask", tempmask);
	//return;
	//pow(src, 1.0f / 3.0, src);
	//imshow("src", src);
	dst = maskH + maskL;
}

Mat zsHDR::maskWeight(uchar maskpixel)
{
	Mat w(LDR_SIZE, 1, CV_32FC1);
	for (int i = 0; i < LDR_SIZE; i++) {
		w.at<float>(i) = i > maskpixel ? 1.0f : 0.0f;
	}
	return w;
}

void zsHDR::Histshow(Mat src, Mat& dst)
{
	int histBinNum = 256;
	float range[] = { 0,256 };
	const float* histRange = { range };
	Mat hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &histBinNum, &histRange);//计算直方图
	uint max_val=src.rows*src.cols;
	//minMaxLoc(hist, 0, &max_val, 0, 0);
	int scale = 2;
	int hist_height = 256;
	dst = Mat::zeros(hist_height, histBinNum*scale, CV_8UC3);
	for (int i = 0; i<histBinNum; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val*hist_height / max_val);  //要绘制的高度
		rectangle(dst, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
}

}
