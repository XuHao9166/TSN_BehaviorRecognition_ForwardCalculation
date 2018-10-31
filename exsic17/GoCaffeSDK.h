#ifndef GO_CAFFE_SDK_H
#define GO_CAFFE_SDK_H

#ifndef SVTCLIENT_EXTERN_C
#ifdef __cplusplus
#define SVTCLIENT_EXTERN_C extern "C"
#else
#define SVTCLIENT_EXTERN_C
#endif
#endif

#define SVTCLIENT_API_EXPORTS /*__declspec(dllimport)*/

#ifndef SVTCLIENT_STDCALL
#if (defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64) && (defined _MSC_VER)
#define SVTCLIENT_STDCALL __stdcall
#else
#define SVTCLIENT_STDCALL
#endif
#endif

#define SVTCLIENT_IMPL SVTCLIENT_EXTERN_C

#define SVTCLIENT_API(rettype) SVTCLIENT_EXTERN_C SVTCLIENT_API_EXPORTS rettype SVTCLIENT_STDCALL


//typedef long long GXXHandle;

//函数返回值
enum GoCaffeErrorCode {
	GO_CAFFE_SUCCEED = 0,         //成功
	GO_CAFFE_INVAILD_PARAM,       //输入参数有误  
	GO_CAFFE_NOT_INIT,            //caffe未初始化
	GO_CAFFE_GETGPUS_ERROR,       //获取GPU信息失败
	GO_CAFFE_NO_GPU,               //不存在GPU
	GO_CAFFE_NO_DEVICEID,         //指定设备不存在
	GO_CAFFE_NET_NOT_INIT,        //现网络未初始化
	GO_CAFFE_NET_INIT_ERROR,      //网络初始化失败：1、网络文件（或权值文件）不存在，2、网络文件和权值文件不对应
	GO_CAFFE_MEAN_ERROR,          //均值文件设置失败：1、均值文件不存在，2、均值文件与网络不对应
	GO_CAFFE_NET_NOCAL,           //未进行前向计算，无法获取指定layer输出  
	GO_CAFFE_NET_CAL_ERROR,       //网络前向计算失败
	GO_CAFFE_NET_GETRESULT_ERROR, //获取指定layer输出失败
	GO_CAFFE_SDK_OUTOF_RES,       //系统资源不足
	GO_CAFFE_ID_NOT_EXIST,         //ID指向实例不存在
	GO_CAFFE_EMPTY_QUEUE          //队列为空
};
//程序应用模式
enum GoCaffeMode {
	GO_CAFFE_CPU_ONLY = 0,         //使用CPU模式
	GO_CAFFE_GPU_SINGLE          //使用单GPU模式，需要指定设备ID，默认使用第一个设备
};

//应用模式信息
typedef struct _GoCaffeModeParams
{
	int nAppMode;              //应用模式
	int nDeviceId;             //设备Id，与应用模式对应
}GoCaffeModeParams;

//初始化参数信息
typedef struct _GoCaffeInitParams
{
	char* szModelFile;     //模型文件
	char* szTrainedFile;   //权值文件
}GoCaffeInitParams;

//初始化参数信息
typedef struct _GoCaffeInitParamsEnc
{
	char* szModelFile;     //加密模型文件
	char* szKey;           //模型解密密码
}GoCaffeInitParamsEnc;

//初始化参数信息
typedef struct _GoCaffeMeanScalar
{
	float fMeanScalar[3];		//均值，对应cv::Scalar([0], [1], [2]）, 灰度输入时只需设置0号位
	float fScale;
}GoCaffeMeanScalar;

#ifdef __cplusplus
extern "C" {
#endif

//////////////////
//得到最近一次的错误码
SVTCLIENT_API(int)          GO_Caffe_GetLastErrorCode(void);
///////////////////
//初始化caffe
SVTCLIENT_API(int)          GO_Caffe_Init();
//清除caffe环境
SVTCLIENT_API(int)          GO_Caffe_CleanUp(void);
//创建caffe实例
SVTCLIENT_API(int)          GO_Caffe_Create(const GoCaffeModeParams* mode_params, const GoCaffeInitParams* init_params, long long &handle);
//创建caffe实例，读取加密模型
SVTCLIENT_API(int)          GO_Caffe_Create_With_Enc(const GoCaffeModeParams* mode_params, const GoCaffeInitParamsEnc* init_params, long long &handle);
//清除caffe实例
SVTCLIENT_API(int)          GO_Caffe_Release(const long long handle);
////////////////////////////////////////////////
// 设置均值文件(建议使用均值代替)
SVTCLIENT_API(int)          GO_Caffe_SetMean(const long long handle, const char* mean_file);
////////////////////////////////////////////////
// 设置调整参数，用于图像预处理
// 入参
// handle          句柄
// fMeanScalar[3]    均值，对单通道数据只取0号位，多三通道取对应012.
//                 另外对于batch_size大于1的只取0号位.
// fScale          缩放因子
SVTCLIENT_API(int)          GO_Caffe_SetCalAdjustment(const long long handle, const GoCaffeMeanScalar* mean_scalar);
////////////////////////////////////////////////
// 网络前向计算
// 入参
// handle          句柄
// img_buffer      图像数据指针，取自cv::Mat
// width           图像宽
// height          图像高
// channels        图像通道数
// batch_size        批量输入个数。在batch_size大于1时，img_buffer的实际
//                 通道数应为channels * batch_size.
//                   如输入5个3通道的图像，那么img_buffer应为15通道的数据，
//                 顺序为图像a通道123、图像b通道123，依次堆叠. 此时入参
//                 channels填3，batch_size为5，SDK内部再做切分.
SVTCLIENT_API(int)          GO_Caffe_CalNet(const long long handle, const char* img_buffer, const int width, const int height, const int channels, const int batch_size);
////////////////////////////////////////////////
// 获取指定layer的输出
// 入参
// handle          句柄
// out_layer_name  输出层的名字
// 出参
// output          输出数据，内存由SDK内部控制
// count           输出数据维度
SVTCLIENT_API(int)          GO_Caffe_GetNetResult(const long long handle, const char* out_layer_name, float** output, int* count);

#ifdef __cplusplus
}
#endif

#endif //GXX_FACE_RECOGNITION_SDK_H