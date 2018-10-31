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



////////////������������ɫϵͳ�����в�ɫͼ����ʾ
void makecolorwheel(vector<Scalar> &colorwheel)  //������������ɫϵͳ
{
	int RY = 15;  //���
	int YG = 6;  //����
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

	//ȷ���˶���Χ��  
	float maxrad = -1;

	//��������������ڱ�׼��fx��fy  
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
			//f = 0; //ȡ��ע���Բ鿴ԭʼɫ�� 

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); //�ı�뾶���ӱ��Ͷ� 
				else
					col *= .75; // ������Χ 
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
	motionToColor(flow, motion2color);//����Ƶ�������Ĺ���ת������ɫ��ʾ
	//resize(flow, flow, Size(340, 256));
	imshow("��ɫ����", motion2color);



	Mat qianjing = Mat::zeros(Size(col, row), CV_8UC3);
	static cv::cuda::GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]); ////�Ժ��������˫ͨ������ͼ����зֽ�
	Mat flowy(planes[1]);



	//resize(opticalFlowOut, opticalFlowOut,Size(col, row));
	drawOptFlowMap_gpu(flowx, flowy, qianjing, 10, CV_RGB(0, 255, 0)); ///������ܹ����õ��Ľ�����ǰ��ͼ��



	/*char outputname[100];
	sprintf(outputname, "../y/y%d.jpg", framenum);*/



	//imwrite(outputname,  flowx);
	//imwrite(outputname, flowy);
	imshow("x", flowx);
	imshow("y", flowy);
	imshow(name, qianjing);  ////��ʾǰ��ͼ��
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
	////�鿴����ĵ÷ֽ��
	/*for (int i = 0; i < count; i++)
	{
	printf("%f, ", scores[i]);
	if (i % 3 == 2)
	printf("\n");
	}*/

	int leibieshu = 3;  ///�������������

	float* result;
	float first_row = 0.0;  ///��һ��
	float second_row = 0.0;
	float third_row = 0.0;

	float Average_first_row = 0.0;  ///��һ��
	float Average_second_row = 0.0;
	float Average_third_row = 0.0;

	int num = 0;
	for (int i = 0; i < count; i++)
	{
		

		if (i % leibieshu == 0) ///ȡ��һ������֮��
		{   
			num++;   ////����ÿһ���ж�����
			first_row += scores[i];
		}
		else if (i % leibieshu == 1) ///ȡ�ڶ�������֮��
		{
			second_row += scores[i];
		}
		else if (i % leibieshu == 2) ///ȡ�ڶ�������֮��
		{
			third_row += scores[i];
		}
			
	}

	//////������ֵ�ľ�ֵ 
	Average_first_row = first_row / num;
	Average_second_row = second_row / num;
	Average_third_row = third_row / num;

	//////��������ֵ����softmax����
	float* yy = new float[leibieshu];  //����һ������ָ�룬�����������ȥ
	yy[0] = Average_first_row;
	yy[1] = Average_second_row;
	yy[2] = Average_third_row;    ///Ŀǰֻ��3���������ֻ��yy[2]

	softmax(yy , leibieshu);  ///���뵽softmax������

	float max = 0;
	int Label = 0;
	for (int i = 0; i < leibieshu; i++)
	{
		if (yy[i]>max) {
			Label = i;
			max = yy[i];
		}
	}

	return Label;  ///0��ʾ������1��ʾ�����˶���2��ʾ�ɵ�
}




int main()
{

///////////���ģ�ͳ�ʼ������	
	// ��ʼ��, ȫ�ֽ�һ��
	int ret = GO_Caffe_Init();
	if (ret != GoCaffeErrorCode::GO_CAFFE_SUCCEED)
	{
		std::cout << "Caffe ��ʼ��ʧ�ܣ�" << std::endl;
		std::cout << "�������Ϊ" << ret << std::endl;
		system("Pause");
		return -1;
	}
	//////////////////////////

	/////����ģ��
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
		std::cout << "Caffe ��ʼ��ʧ�ܣ�" << std::endl;
		std::cout << "�������Ϊ" << ret << std::endl;
		system("Pause");
		return 0;
	}
/////////////////////////////////////////////////////



///////////////������ܹ�����ⲿ��
	int s = 1;
	float  scores=0;
	float result_second=0;
	vector<float> scores_second;
	int test_time = 3; // ����ʱ�䳤�� / s  int
	bool save_state = false; // �����Ƿ񱣴�ָ������������Ƶ��

	unsigned long AAtime = 0, BBtime = 0;
	double Bound = 20;
	
	Mat GetImg, next, prvs;
	Mat anti_img_x;
	Mat anti_img_y;
	
	vector<Mat> channels;
	vector<Mat> src_channels;
	vector<Mat> enforce_channels;

	//gpu ����
	cv::cuda::GpuMat prvs_gpu, next_gpu, flow_gpu;
	cv::cuda::GpuMat prvs_gpu_o, next_gpu_o;
	cv::cuda::GpuMat prvs_gpu_c, next_gpu_c;
	static cv::cuda::GpuMat discrete[2];

	/////��������ͼ����
	//GetImg = imread("../testImage/0.jpg");
	////////////

	///////////////����Ƶ��
	VideoCapture cap("20180718_093822 (2).avi");     
	//VideoCapture cap(0);
	if (!(cap.read(GetImg))) //��ȡ��Ƶ����֡
		return 0;
	//////////////////////

	resize(GetImg, GetImg, Size(340, 256));
	prvs_gpu_o.upload(GetImg);
	cv::cuda::resize(prvs_gpu_o, prvs_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
	cv::cuda::cvtColor(prvs_gpu_c, prvs_gpu, CV_BGR2GRAY);

	//���ܹ������
	
    // static Ptr<cuda::FarnebackOpticalFlow> fbOF = cuda::FarnebackOpticalFlow::create();
	//static Ptr<cuda::OpticalFlowDual_TVL1> fbOF = cuda::OpticalFlowDual_TVL1::create();
	static Ptr<cuda::OpticalFlowDual_TVL1> fbOF = cuda::OpticalFlowDual_TVL1::create();
	int framenum = 0;
/////////////////////////////////////////////////////////////////////////////////////////////

///////////////////����������ǿģ�岿��
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


		/////��������ͼ����
	/*	sprintf(filename, "../testImage/%d.jpg", framenum);
		GetImg = imread(filename);*/
		////////////

		/////����Ƶ��
		if (!(cap.read(GetImg))) 
			break;
		/////////////

		resize(GetImg, GetImg, Size(340,256));

		imshow("src", GetImg);
		////Mat ת GpuMat 
		next_gpu_o.upload(GetImg);
		cv::cuda::resize(next_gpu_o, next_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
		cv::cuda::cvtColor(next_gpu_c, next_gpu, CV_BGR2GRAY);
		///////////////////////////////////////////////////////////////////

		AAtime = getTickCount();
		//////���г��ܹ���������
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
		
		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		//printf("%.2lf / %.2lf \n", pt, fpt);
		////////////////////////////

		/////������õ�����������ֽ��x,y��������Ĺ���	
		cuda::split(flow_gpu, discrete);

		Mat flow_x(discrete[0]); ////�Ժ��������˫ͨ������ͼ����зֽ�
		Mat flow_y(discrete[1]);
		//////////////

		/////////////��һ���������ӻ�
		Mat img_x(flow_x.size(), CV_8UC1); ///������붨��Ϊ��ͨ��
		Mat img_y(flow_y.size(), CV_8UC1);
		convertFlowToImage(flow_x, flow_y, img_x, img_y, -Bound, Bound);

		/*imshow("img_x",img_x);
		imshow("img_y", img_y);*/

		/////��img_x��img_y���о��洹ֱ��ת  (��ԭ��д�����Ǿ���ȡ���󣬺�������ü�������ʵ�������ǲü�������ľ���)
		//flip(img_x, anti_img_x, 1);
		//flip(img_y, anti_img_y, 1);
		////////��ֱ��ת֮��������صķ���任 -��255-��ǰ����
		//anti_img_x = ~anti_img_x;
		//anti_img_y = ~anti_img_y;
		////////////////////////////////


		////5��ԭ֡->10�ŵ�ͨ�����ܹ���ͼ->10*10��ÿ����ǿ10�Σ�=100 �� ��ͨ�����ͼ
		//1��ԭ֡->2�ŵ�ͨ�����ܹ���ͼ->2*10��ÿ����ǿ10�ΰ���������ROI5�κʹ�ֱ��תȡ����ROI5�Σ�=20 �� ��ͨ�����ͼ
		//if (channels.size() < 100)
		//{
			//////����������ǿѹ��ȥ
			/////��֮����������forѭ��ѹ��ȥ������Ϊ��100ͨ��������ÿ10ͨ����Ϊһ�鰴˳����������
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

			//	/////�ü����ͼ����������ֱ��ת
			//	flip(img_x(ROI[i]), anti_img_x, 1);
			//	flip(img_y(ROI[i]), anti_img_y, 1);
			//	////��ֱ��ת֮��������صķ���任 -��255-��ǰ����
			//	anti_img_x = ~anti_img_x;
			//	anti_img_y = ~anti_img_y;
			//	////���õ��Ľ��ѹ��channels
			//	channels.push_back(anti_img_x);
			//	channels.push_back(anti_img_y);

			//}




	//	}


		if (src_channels.size() < 10)
		{
		/*	sprintf(filename1, "../test/%d.jpg", framenum);
			img_x = imread(filename1,0);*/

			src_channels.push_back(img_x); ///������5֡������10�Ź���ͼѹ��ȥ
			src_channels.push_back(img_y);

			if (src_channels.size() == 10)
			{
				for (int j = 0; j < 5; j++)  ////�ֳ�5�飬����ÿһ����10�Ź���ͳһ�����Ͻ�224*224 ��ROI
				{
					for (int i = 0; i < 10; i++)
					{
						Mat anti_img;
						Mat midel;
					    midel = src_channels[i](ROI[j]);
						//resize(src_channels[i], midel, Size(224, 224));
						
							
						///������ֱ��ת
						flip(midel, anti_img, 1);
						////��ֱ��ת֮��������صķ���任 -��255-��ǰ����
						anti_img = ~anti_img;
						////���õ��Ľ��ѹ��enforce_channels
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
			merge(channels, src);  ///��100�ŵ�ͨ��ͼƬ�ϲ���һ��100ͨ����ͼƬ

			channels.clear(); ////�������������ɾ�		
			src_channels.clear();
			enforce_channels.clear();

			///ǰ�����
			time_t stime = clock();

			double tt1 = cvGetTickCount();

			////��һ��100ͨ���������ݣ�ÿʮ��Ϊһ��
			ret = GO_Caffe_CalNet(handle, (const char*)src.data, src.cols, src.rows, 10, 10); ///���һ��10��ʾbatchΪ10,�����ڶ���10��ʾÿ��batch��channels��Ϊ10

			tt1 = cvGetTickCount() - tt1;
			tt1 = tt1 / (cvGetTickFrequency() * 1000);
			//printf("ǰ�����ʱ��:ms %f\n", tt1);
			///////////////
			////////////��ȡָ������//////////////////////////////////////////////////////
			std::string outLayerName = "fc-action";
			float* scores = NULL;
			int count = 0;
			tt1 = cvGetTickCount();
			ret = GO_Caffe_GetNetResult(handle, outLayerName.c_str(), &scores, &count);  ///outputΪ����ĵ÷֣�count��ʾά��
			tt1 = cvGetTickCount() - tt1;
			tt1 = tt1 / (cvGetTickFrequency() * 1000);
			//printf("��ȡָ��layerʱ��:ms %f\n", tt1);
			
			//printf("count = %d.\n", count);
			/////////////////////////

			//////�鿴����ĵ÷ֽ��
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

		

	 //  /////�Ұ���ʾ����ȫ��������һ��ͬʱ����
		//showFlow("OpticalFlowFarneback", flow_gpu, (GetImg.size().width / s), (GetImg.size().height / s), framenum);
		/////////////////////////////////////////////////////////////////////

		//////���³��ܹ�����������
		prvs_gpu = next_gpu.clone();

		if (waitKey(5) >= 0)
		{
			break;
		}
			
	}

	GO_Caffe_Release(handle);
	GO_Caffe_CleanUp();
}