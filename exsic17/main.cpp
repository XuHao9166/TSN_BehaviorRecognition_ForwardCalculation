#include < stdio.h>
#include < iostream>

#include < opencv2\opencv.hpp>
#include < opencv2/core/core.hpp>
#include < opencv2/highgui/highgui.hpp>
#include < opencv2/video/background_segm.hpp>


#include <cuda_runtime.h>  
#include <cuda.h> 

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp> 
#include <opencv2/cudalegacy.hpp>

#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp> 

#include "GoCaffeSDK.h"
#include <iostream>
#include "io.h"
#include "opencv2/opencv.hpp"
#include <thread>

using namespace cv;
using namespace std;

typedef long long Handle;
//#define ENC
#define UNKNOWN_FLOW_THRESH 1e9  



////////////定义孟塞尔颜色系统来进行彩色图像显示
void makecolorwheel(vector<Scalar> &colorwheel)  //定义孟塞尔颜色系统
{
	int RY = 15;  //红黄
	int YG = 6;  //黄绿
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	//确定运动范围：  
	float maxrad = -1;

	//查找最大流量用于标准化fx和fy  
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; //取消注释以查看原始色轮 

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); //改变半径增加饱和度 
				else
					col *= .75; // 超出范围 
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}
/////////////////////////////////////////


void drawOptFlowMap_gpu(const Mat& flow_x, const Mat& flow_y, Mat& cflowmap, int step, const Scalar& color) {



	for (int y = 0; y < cflowmap.rows; y += step)
	for (int x = 0; x < cflowmap.cols; x += step)
	{
		Point2f fxy;
		fxy.x = cvRound(flow_x.at< float >(y, x) + x);
		fxy.y = cvRound(flow_y.at< float >(y, x) + y);

		line(cflowmap, Point(x, y), Point(fxy.x, fxy.y), color);
		circle(cflowmap, Point(fxy.x, fxy.y), 1, color, -1);
	}
}

void showFlow(const char* name, const cv::cuda::GpuMat& d_flow, int col, int row, int framenum)
{

	Mat motion2color;
	Mat flow(d_flow);
	motionToColor(flow, motion2color);//将视频检测出来的光流转化成颜色显示
	//resize(flow, flow, Size(340, 256));
	imshow("彩色光流", motion2color);



	Mat qianjing = Mat::zeros(Size(col, row), CV_8UC3);
	static cv::cuda::GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]); ////对函数输出的双通道光流图像进行分解
	Mat flowy(planes[1]);



	//resize(opticalFlowOut, opticalFlowOut,Size(col, row));
	drawOptFlowMap_gpu(flowx, flowy, qianjing, 10, CV_RGB(0, 255, 0)); ///输入稠密光流得到的结果输出前景图像



	/*char outputname[100];
	sprintf(outputname, "../y/y%d.jpg", framenum);*/



	//imwrite(outputname,  flowx);
	//imwrite(outputname, flowy);
	imshow("x", flowx);
	imshow("y", flowy);
	imshow(name, qianjing);  ////显示前景图像
	//waitKey(3);
}




void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
	double lowerBound, double higherBound) 
{
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i, j);
			float y = flow_y.at<float>(i, j);
			img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i, j) = CAST(y, lowerBound, higherBound);
		}
	}
#undef CAST
}

void softmax(float *x , int k)
{
	float max = 0.0;
	float sum = 0.0;

	for (int i = 0; i<k; i++) if (max < x[i]) max = x[i];
	for (int i = 0; i<k; i++) {
		x[i] = exp(x[i] - max);
		sum += x[i];
	}

	for (int i = 0; i<k; i++) x[i] /= sum;
}


int Average_softmax(float* scores, int count)
{
	////查看输出的得分结果
	/*for (int i = 0; i < count; i++)
	{
	printf("%f, ", scores[i]);
	if (i % 3 == 2)
	printf("\n");
	}*/

	int leibieshu = 3;  ///网络分类的类别数

	float* result;
	float first_row = 0.0;  ///第一列
	float second_row = 0.0;
	float third_row = 0.0;

	float Average_first_row = 0.0;  ///第一列
	float Average_second_row = 0.0;
	float Average_third_row = 0.0;

	int num = 0;
	for (int i = 0; i < count; i++)
	{
		

		if (i % leibieshu == 0) ///取第一列数字之和
		{   
			num++;   ////计算每一列有多少行
			first_row += scores[i];
		}
		else if (i % leibieshu == 1) ///取第二列数字之和
		{
			second_row += scores[i];
		}
		else if (i % leibieshu == 2) ///取第二列数字之和
		{
			third_row += scores[i];
		}
			
	}

	//////求三个值的均值 
	Average_first_row = first_row / num;
	Average_second_row = second_row / num;
	Average_third_row = third_row / num;

	//////将三个均值输入softmax函数
	float* yy = new float[leibieshu];  //定义一个数组指针，将三个数存进去
	yy[0] = Average_first_row;
	yy[1] = Average_second_row;
	yy[2] = Average_third_row;    ///目前只分3个类别。所以只到yy[2]

	softmax(yy , leibieshu);  ///输入到softmax函数中

	float max = 0;
	int Label = 0;
	for (int i = 0; i < leibieshu; i++)
	{
		if (yy[i]>max) {
			Label = i;
			max = yy[i];
		}
	}

	return Label;  ///0表示正常；1表示激烈运动；2表示躺地
}




int main()
{

///////////深度模型初始化部分	
	// 初始化, 全局仅一次
	int ret = GO_Caffe_Init();
	if (ret != GoCaffeErrorCode::GO_CAFFE_SUCCEED)
	{
		std::cout << "Caffe 初始化失败！" << std::endl;
		std::cout << "错误编码为" << ret << std::endl;
		system("Pause");
		return -1;
	}
	//////////////////////////

	/////加载模型
	long long handle = -1;
	GoCaffeModeParams mode_params;
	mode_params.nAppMode = GoCaffeMode::GO_CAFFE_GPU_SINGLE;
	mode_params.nDeviceId = 0;
#ifndef ENC
	GoCaffeInitParams init_params;
	std::string proto_txt = "../model/tsn_bn_inception_flow_deploy_no_dropout.prototxt";
	std::string caffe_model = "../model/myvideos3_split1_tsn_flow_bn_inception_iter_5000.caffemodel";
	init_params.szModelFile = (char*)proto_txt.c_str();
	init_params.szTrainedFile = (char*)caffe_model.c_str();
	 ret = GO_Caffe_Create(&mode_params, &init_params, handle);

#else
	std::string bin_model = "C:\\Users\\chenjianming\\Desktop\\model_enc_test\\model\\vvv.gcnn_success";
	std::string model_key = "654321";
	GoCaffeInitParamsEnc initParameter;
	initParameter.szModelFile = (char*)bin_model.c_str();
	initParameter.szKey = (char*)model_key.c_str();
	int ret = GO_Caffe_Create_With_Enc(&mode_params, &initParameter, 1, handle);
#endif

	GoCaffeMeanScalar mean_scalar;
	mean_scalar.fScale = 1;
	mean_scalar.fMeanScalar[0] = -128.0;
	mean_scalar.fMeanScalar[1] = 0;
	mean_scalar.fMeanScalar[2] = 0;
	GO_Caffe_SetCalAdjustment(handle, &mean_scalar);
	if (ret != GoCaffeErrorCode::GO_CAFFE_SUCCEED)
	{
		std::cout << "Caffe 初始化失败！" << std::endl;
		std::cout << "错误编码为" << ret << std::endl;
		system("Pause");
		return 0;
	}
/////////////////////////////////////////////////////



///////////////定义稠密光流检测部分
	int s = 1;
	float  scores=0;
	float result_second=0;
	vector<float> scores_second;
	int test_time = 3; // 测试时间长度 / s  int
	bool save_state = false; // 设置是否保存指定报警动作视频段

	unsigned long AAtime = 0, BBtime = 0;
	double Bound = 20;
	
	Mat GetImg, next, prvs;
	Mat anti_img_x;
	Mat anti_img_y;
	
	vector<Mat> channels;
	vector<Mat> src_channels;
	vector<Mat> enforce_channels;

	//gpu 变量
	cv::cuda::GpuMat prvs_gpu, next_gpu, flow_gpu;
	cv::cuda::GpuMat prvs_gpu_o, next_gpu_o;
	cv::cuda::GpuMat prvs_gpu_c, next_gpu_c;
	static cv::cuda::GpuMat discrete[2];

	/////批量测试图像用
	//GetImg = imread("../testImage/0.jpg");
	////////////

	///////////////跑视频用
	VideoCapture cap("20180718_093822 (2).avi");     
	//VideoCapture cap(0);
	if (!(cap.read(GetImg))) //获取视频的首帧
		return 0;
	//////////////////////

	resize(GetImg, GetImg, Size(340, 256));
	prvs_gpu_o.upload(GetImg);
	cv::cuda::resize(prvs_gpu_o, prvs_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
	cv::cuda::cvtColor(prvs_gpu_c, prvs_gpu, CV_BGR2GRAY);

	//稠密光流检测
	
    // static Ptr<cuda::FarnebackOpticalFlow> fbOF = cuda::FarnebackOpticalFlow::create();
	//static Ptr<cuda::OpticalFlowDual_TVL1> fbOF = cuda::OpticalFlowDual_TVL1::create();
	static Ptr<cuda::OpticalFlowDual_TVL1> fbOF = cuda::OpticalFlowDual_TVL1::create();
	int framenum = 0;
/////////////////////////////////////////////////////////////////////////////////////////////

///////////////////定义数据增强模板部分
	vector<Rect> ROI;
	// GetImg.rows = 256
	Rect rect1 = Rect(0, 0, 224, 224);
	Rect rect2 = Rect(0, GetImg.rows-224 , 224, 224);
	Rect rect3 = Rect(GetImg.cols-224 , 0, 224, 224);
	Rect rect4 = Rect(GetImg.cols - 224, GetImg.rows - 224, 224, 224);
	Rect rect5 = Rect(0.5*(GetImg.cols - 224), 0.5*(GetImg.rows - 224), 224, 224);

	ROI.push_back(rect1);
	ROI.push_back(rect2);
	ROI.push_back(rect3);
	ROI.push_back(rect4);
	ROI.push_back(rect5);
/////////////////////////////////////



	channels.clear();
	src_channels.clear();
	enforce_channels.clear();
	char filename[50];
	char filename1[50];
	while (true) {

		framenum++;
		//cout << framenum << endl;


		/////批量测试图像用
	/*	sprintf(filename, "../testImage/%d.jpg", framenum);
		GetImg = imread(filename);*/
		////////////

		/////跑视频用
		if (!(cap.read(GetImg))) 
			break;
		/////////////

		resize(GetImg, GetImg, Size(340,256));

		imshow("src", GetImg);
		////Mat 转 GpuMat 
		next_gpu_o.upload(GetImg);
		cv::cuda::resize(next_gpu_o, next_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
		cv::cuda::cvtColor(next_gpu_c, next_gpu, CV_BGR2GRAY);
		///////////////////////////////////////////////////////////////////

		AAtime = getTickCount();
		//////进行稠密光流检测计算
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
		
		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		//printf("%.2lf / %.2lf \n", pt, fpt);
		////////////////////////////

		/////将计算得到的整体光流分解成x,y两个方向的光流	
		cuda::split(flow_gpu, discrete);

		Mat flow_x(discrete[0]); ////对函数输出的双通道光流图像进行分解
		Mat flow_y(discrete[1]);
		//////////////

		/////////////归一化光流可视化
		Mat img_x(flow_x.size(), CV_8UC1); ///输出必须定义为单通道
		Mat img_y(flow_y.size(), CV_8UC1);
		convertFlowToImage(flow_x, flow_y, img_x, img_y, -Bound, Bound);

		/*imshow("img_x",img_x);
		imshow("img_y", img_y);*/

		/////对img_x和img_y进行镜面垂直翻转  (我原来写这里是镜像取反后，后面才做裁剪。。而实际论文是裁剪后才做的镜像)
		//flip(img_x, anti_img_x, 1);
		//flip(img_y, anti_img_y, 1);
		////////垂直翻转之后进行像素的反向变换 -》255-当前像素
		//anti_img_x = ~anti_img_x;
		//anti_img_y = ~anti_img_y;
		////////////////////////////////


		////5张原帧->10张单通道稠密光流图->10*10（每张增强10次）=100 张 单通道结果图
		//1张原帧->2张单通道稠密光流图->2*10（每张增强10次包括：正面ROI5次和垂直翻转取反后ROI5次）=20 张 单通道结果图
		//if (channels.size() < 100)
		//{
			//////增加数据增强压进去
			/////我之所以用两个for循环压进去，是因为这100通道的数据每10通道归为一组按顺序排下来的
			//for (int j = 0; j < 5; j++)
			//{
			//	channels.push_back(img_x(ROI[j]));
			//	channels.push_back(img_y(ROI[j]));
			//	
			//}

			//for (int i = 0; i < 5; i++)
			//{
			//	/*channels.push_back(anti_img_x(ROI[i]));
			//	channels.push_back(anti_img_y(ROI[i]));*/

			//	/////裁剪后的图像，再做镜像垂直翻转
			//	flip(img_x(ROI[i]), anti_img_x, 1);
			//	flip(img_y(ROI[i]), anti_img_y, 1);
			//	////垂直翻转之后进行像素的反向变换 -》255-当前像素
			//	anti_img_x = ~anti_img_x;
			//	anti_img_y = ~anti_img_y;
			//	////将得到的结果压入channels
			//	channels.push_back(anti_img_x);
			//	channels.push_back(anti_img_y);

			//}




	//	}


		if (src_channels.size() < 10)
		{
		/*	sprintf(filename1, "../test/%d.jpg", framenum);
			img_x = imread(filename1,0);*/

			src_channels.push_back(img_x); ///把连续5帧产生的10张光流图压进去
			src_channels.push_back(img_y);

			if (src_channels.size() == 10)
			{
				for (int j = 0; j < 5; j++)  ////分成5组，例：每一组是10张光流统一的左上角224*224 的ROI
				{
					for (int i = 0; i < 10; i++)
					{
						Mat anti_img;
						Mat midel;
					    midel = src_channels[i](ROI[j]);
						//resize(src_channels[i], midel, Size(224, 224));
						
							
						///做镜像垂直翻转
						flip(midel, anti_img, 1);
						////垂直翻转之后进行像素的反向变换 -》255-当前像素
						anti_img = ~anti_img;
						////将得到的结果压入enforce_channels
						enforce_channels.push_back(anti_img);		

						channels.push_back(midel);
					}
				}

				for (int k = 0; k < enforce_channels.size(); k++)
				{
					
					channels.push_back(enforce_channels[k]);

				}
				

			}


		}
		else
		{    
			Mat src;
			merge(channels, src);  ///将100张单通道图片合并成一张100通道的图片

			channels.clear(); ////用完把容器清理干净		
			src_channels.clear();
			enforce_channels.clear();

			///前向计算
			time_t stime = clock();

			double tt1 = cvGetTickCount();

			////传一串100通道长的数据，每十个为一组
			ret = GO_Caffe_CalNet(handle, (const char*)src.data, src.cols, src.rows, 10, 10); ///最后一个10表示batch为10,倒数第二个10表示每个batch的channels数为10

			tt1 = cvGetTickCount() - tt1;
			tt1 = tt1 / (cvGetTickFrequency() * 1000);
			//printf("前向计算时间:ms %f\n", tt1);
			///////////////
			////////////获取指定层结果//////////////////////////////////////////////////////
			std::string outLayerName = "fc-action";
			float* scores = NULL;
			int count = 0;
			tt1 = cvGetTickCount();
			ret = GO_Caffe_GetNetResult(handle, outLayerName.c_str(), &scores, &count);  ///output为输出的得分，count表示维度
			tt1 = cvGetTickCount() - tt1;
			tt1 = tt1 / (cvGetTickFrequency() * 1000);
			//printf("获取指定layer时间:ms %f\n", tt1);
			
			//printf("count = %d.\n", count);
			/////////////////////////

			//////查看输出的得分结果
			//for (int i = 0; i < count; i++)
			//{
			//	printf("%f, ", scores[i]);
			//	if (i % 3 == 2)
			//		printf("\n");
			//}
			//if ()
			int result_label = Average_softmax(scores, count);

			cout << result_label << endl;
		}

		

	 //  /////我把显示函数全都集中在一起同时绘制
		//showFlow("OpticalFlowFarneback", flow_gpu, (GetImg.size().width / s), (GetImg.size().height / s), framenum);
		/////////////////////////////////////////////////////////////////////

		//////更新稠密光流计算输入
		prvs_gpu = next_gpu.clone();

		if (waitKey(5) >= 0)
		{
			break;
		}
			
	}

	GO_Caffe_Release(handle);
	GO_Caffe_CleanUp();
}