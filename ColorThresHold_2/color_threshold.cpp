#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdlib.h>
#include "unistd.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

//Position class 
class Position
{
public:
    float X;
    float Y;
    float Z;
    float pitch;
    float roll;
    float yaw;
    float timestamp;
};

//GBR Color tuner
void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);

//HSV Color tuner
void on_low_h_thresh_trackbar(int, void *);
void on_high_h_thresh_trackbar(int, void *);
void on_low_s_thresh_trackbar(int, void *);
void on_high_s_thresh_trackbar(int, void *);
void on_low_v_thresh_trackbar(int, void *);
void on_high_v_thresh_trackbar(int, void *);

// Read frame data --> IMU/GPS data
void readFrameData(const string &strFrameDataFile, vector<double> &vTimestamps, vector<string> &vFrameNames, vector<Position> &vPositions);

Position pastPosition(vector<Position> &vPositions, float timestamp);

// Read config file -> Kmatrix, Distortion matrix and so on.
bool readConfig(const string &strConfigFile, Mat &Kmatrix, Mat &DisMatrix, vector<float> &Dvector);

//Initial GBR color tuner
int low_r=20, low_g=20, low_b=10;
int high_r=88, high_g=77, high_b=136;

//Initial HSV color tuner
int low_h =0, low_s = 0, low_v = 0;
int high_h = 100, high_s = 100, high_v = 100;

//Image path
const string img_path = "/home/teeramoo/Desktop/ORB-slam-script/Using Opecv3.2/Line Segment Detector/Sequence_images_color/color00000452.png";

//Video path
const string vid_path = "/home/teeramoo/Desktop/11May2017-Survey/2017-05-11_13_29_49.mkv";

int main(int argc, char **argv)
{

//Check arguments
if(argc != 4)
    {
        cout << "usage : ./color_threshold <path_to_video_file> <path_to_IMU_log_file> <path_to_camera_setting_file>" << endl;
        return 0;
    }

//Variable declaration
    vector<Position> vPositions;
    vector<string> vFramenames;
    vector<double> vTimestamps;
    Mat composed_image, front_image, down_image;
    Mat frame, frameHSV, frameBGR, frame_thresholdHSV, frame_threshold, frame_GRAY;

    //Variable for config file    
    Mat KmatrixCV32_F = Mat(3,3, CV_32F);
    Mat DisMatrix = Mat(5,1,CV_32F);
    vector<float> Dvector ;

    int nFrames;
   
//Read arguments
    string video_path = argv[1];
    string logfile_path = argv[2];
    string camera_settings = argv[3];


    VideoCapture cap(vid_path);
    if(!cap.isOpened())
    	{
        cout << "Cannot read video file." << endl;
        return 0;
    	}

//Set window name
    namedWindow("Video Capture", WINDOW_NORMAL);
    namedWindow("RGB Detection", WINDOW_NORMAL);
    namedWindow("HSV Detection", WINDOW_NORMAL);
    //-- Trackbars to set thresholds for RGB values
    createTrackbar("Low R","RGB Detection", &low_r, 255, on_low_r_thresh_trackbar);
    createTrackbar("High R","RGB Detection", &high_r, 255, on_high_r_thresh_trackbar);
    createTrackbar("Low G","RGB Detection", &low_g, 255, on_low_g_thresh_trackbar);
    createTrackbar("High G","RGB Detection", &high_g, 255, on_high_g_thresh_trackbar);
    createTrackbar("Low B","RGB Detection", &low_b, 255, on_low_b_thresh_trackbar);
    createTrackbar("High B","RGB Detection", &high_b, 255, on_high_b_thresh_trackbar);

    //-- Trackbars to set thresholds for HSV values
    createTrackbar("Low H","HSV Detection", &low_h, 179, on_low_h_thresh_trackbar);
    createTrackbar("High H","HSV Detection", &high_h, 179, on_high_h_thresh_trackbar);
    createTrackbar("Low S","HSV Detection", &low_s, 255, on_low_s_thresh_trackbar);
    createTrackbar("High S","HSV Detection", &high_s, 255, on_high_s_thresh_trackbar);
    createTrackbar("Low V","HSV Detection", &low_v, 255, on_low_v_thresh_trackbar);
    createTrackbar("High V","HSV Detection", &high_v, 255, on_high_v_thresh_trackbar);

// Read frame data
    readFrameData(logfile_path,vTimestamps,vFramenames,vPositions);	

// Read config data
    if(!readConfig(camera_settings,KmatrixCV32_F,DisMatrix,Dvector))
    {
        cout << "Cannot read setting file." << endl;
        return 0;
    }

    nFrames = vFramenames.size();

//Main loop

	for(int i = 0; i< nFrames; i++)
	{
	//Read image	
	cap.read(composed_image);
	
	//Get downward images
	cv::Mat matFrameDownwardRoi(composed_image, cv::Rect(0, composed_image.rows / 2, composed_image.cols, composed_image.rows / 2));

        matFrameDownwardRoi.copyTo(down_image);

	//Get forward images
        cv::Mat matFrameForwardRoi(composed_image, cv::Rect(0, 0, composed_image.cols, composed_image.rows / 2));

        matFrameForwardRoi.copyTo(front_image);	
	
	//Undistort front image
	cv::undistort(front_image,frame,KmatrixCV32_F,Dvector);	

	cvtColor(frame, frameHSV, CV_BGR2HSV);	
	inRange(frameHSV,Scalar(low_h,low_s,low_v), Scalar(high_h,high_s,high_v),frame_thresholdHSV);
	
//	cvtColor(frame_thresholdHSV, frameBGR, CV_HSV2BGR);
	
        //-- Detect the object based on RGB Range Values
       inRange(frame,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),frame_threshold);
        

//Line segment detector
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);

        cvtColor(frame, frame_GRAY, COLOR_RGB2GRAY);
        double start = double(getTickCount());
        vector<Vec4f> lines_std_GBR, lines_std_HSV, lines_std_GRAY;
        vector<int> lines_width_GBR, lines_width_HSV, lines_width_GRAY;
	
	//Detect
	ls->detect(frame_threshold, lines_std_GBR, lines_width_GBR);
	ls->detect(frame_thresholdHSV, lines_std_HSV, lines_width_HSV);
	ls->detect(frame_GRAY, lines_std_GRAY, lines_width_GRAY);

	//Draw Lines
	Mat drawnLinesGBR(frame_threshold);
	Mat drawnLinesHSV(frame_thresholdHSV);
	Mat drawnLinesGRAY(frame_GRAY);

	//Draw segment
        ls->drawSegments(drawnLinesGBR, lines_std_GBR);
        ls->drawSegments(drawnLinesHSV, lines_std_HSV);
        ls->drawSegments(drawnLinesGRAY, lines_std_GRAY);


        double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "It took " << duration_ms << " ms." << std::endl;


//-- Show the frames
//        imshow("Video Capture",frame);
//        imshow("RGB Detection",frame_threshold);
//	imshow("HSV Detection",frame_thresholdHSV);
	
	imshow("Video Capture",frame);
        imshow("RGB Detection",drawnLinesGBR);
	imshow("HSV Detection",drawnLinesHSV);

	namedWindow("Standard refinement", WINDOW_NORMAL);
        imshow("GRAY Detection", drawnLinesGRAY);

	waitKey(100);
    }
    return 0;
}

void tokenize(const string &str, vector<string> &vTokens)
{
    int iPos = 0;
    int iTokBeg = 0;
    while (iPos < (int) str.length())
    {
        if (str[iPos] == ' ')
        {
            if (iTokBeg < iPos)
            {
                vTokens.push_back(str.substr(iTokBeg, iPos - iTokBeg));
                iTokBeg = iPos + 1;
            }
        }
        iPos++;
    }
    if (iTokBeg < (int) str.length())
        vTokens.push_back(str.substr(iTokBeg));
}

void readFrameData(const string &strFrameDataFile,  vector<double> &vTimestamps, vector<string> &vFrameNames, vector<Position> &vPositions)
{
    ifstream f;
    f.open(strFrameDataFile.c_str());
    if (!f.is_open())
    {
        ostringstream ostream;
        ostream << "Could not open frame data file " << strFrameDataFile;
        throw runtime_error(ostream.str());
    }

    bool bFirstGps = false;
    int latOr = 0, lonOr = 0, altOr = 0;
    Position positionLatest;
    positionLatest.X = 0;
    positionLatest.Y = 0;
    positionLatest.Z = 0;
    positionLatest.roll = 0;
    positionLatest.pitch = 0;
    positionLatest.yaw = 0;
    positionLatest.timestamp = 0;
    vector<Position> vPositionsAll;
    while (f.is_open() && !f.eof())
    {
        string s;
        getline(f, s);
        if (s.length() == 0)
            continue;
        if (s[0] == '#')
            continue;
        vector<string> vTokens;
        tokenize(s, vTokens);
        if (vTokens[0] == "Output")
        {
            // Frame descriptor
            vTimestamps.push_back(std::stod(vTokens[8]));
            vFrameNames.push_back(vTokens[2]);
            if (vPositionsAll.size() > 0)
            {
                vPositions.push_back(
                        pastPosition(vPositionsAll, positionLatest.timestamp));
            }
            else
            {
                vPositions.push_back(positionLatest);
            }
        }
        else if (vTokens[1] == "lat" && vTokens[3] == "lon")
        {
            // GLOBAL_POSITION_INT
            positionLatest.timestamp = std::stof(vTokens[0]);
            int lat = std::stod(vTokens[2]);
            int lon = std::stod(vTokens[4]);
            int alt = std::stod(vTokens[6]);
#if 0
            int relalt = std::stod(vTokens[8]);
            int vx = std::stod(vTokens[10]);
            int vy = std::stod(vTokens[12]);
            int vz = std::stod(vTokens[14]);
            int hdg = std::stod(vTokens[16]);
#endif
            if (!bFirstGps)
            {
                latOr = lat;
                lonOr = lon;
                altOr = alt;
                bFirstGps = true;
            }
            vPositionsAll.push_back(positionLatest);
        }
        else if (vTokens[1] == "roll" && vTokens[3] == "pitch")
        {
            // ATTITUDE
            positionLatest.roll = std::stof(vTokens[2]);
            positionLatest.pitch = std::stof(vTokens[4]);
            positionLatest.yaw = std::stof(vTokens[6]);
        }
    }
}
#define POSITION_FREQ 150
Position pastPosition(vector<Position> &vPositions, float timestamp)
{
    int n = (int) vPositions.size();
    float timestampLast = vPositions[n - 1].timestamp;
    float timestampDiff = timestampLast - timestamp;
    int i = n - 1 - (int) (timestampDiff * POSITION_FREQ);
    if (i < 0)
        i = 0;
    if (vPositions[i].timestamp < timestamp)
    {
        while (i < n && vPositions[i].timestamp < timestamp)
            i++;
        if (i >= n)
            i = n - 1;
    }
    else
    {
        while (i >= 0 && vPositions[i].timestamp >= timestamp)
            i--;
        if (i < 0)
            i = 0;
        else if (i < n - 1)
            i++;
    }
    return vPositions[i];
}

bool readConfig(const string &strConfigFile, Mat &Kmatrix, Mat &DisMatrix, vector<float> &Dvector)
{

    FileStorage setting_file(strConfigFile,FileStorage::READ);
    if(!setting_file.isOpened())
    {
        return false;
    }

    setting_file["camera_matrix_forward"] >> Kmatrix;
    setting_file["distortion_coefficients_forward"] >> DisMatrix;

    for(int i = 0; i < DisMatrix.rows ; i++)
    {
        Dvector.push_back(DisMatrix.at<double>(i,0));
    }

    cout << "Camera matrix" << Kmatrix << endl;
    cout << "DisMatrix" << DisMatrix << endl;

    return true;
}


void on_low_r_thresh_trackbar(int, void *)
{
    low_r = min(high_r-1, low_r);
    setTrackbarPos("Low R","RGB Detection", low_r);
}
void on_high_r_thresh_trackbar(int, void *)
{
    high_r = max(high_r, low_r+1);
    setTrackbarPos("High R", "RGB Detection", high_r);
}
void on_low_g_thresh_trackbar(int, void *)
{
    low_g = min(high_g-1, low_g);
    setTrackbarPos("Low G","RGB Detection", low_g);
}
void on_high_g_thresh_trackbar(int, void *)
{
    high_g = max(high_g, low_g+1);
    setTrackbarPos("High G", "RGB Detection", high_g);
}
void on_low_b_thresh_trackbar(int, void *)
{
    low_b= min(high_b-1, low_b);
    setTrackbarPos("Low B","RGB Detection", low_b);
}
void on_high_b_thresh_trackbar(int, void *)
{
    high_b = max(high_b, low_b+1);
    setTrackbarPos("High B", "RGB Detection", high_b);
}

void on_low_h_thresh_trackbar(int, void *)
{
    low_h = min(high_h-1, low_h);
    setTrackbarPos("Low H","HSV Detection", low_h);
}
void on_high_h_thresh_trackbar(int, void *)
{
    high_h = max(high_h, low_h+1);
    setTrackbarPos("High H", "HSV Detection", high_h);
}
void on_low_s_thresh_trackbar(int, void *)
{
    low_s = min(high_s-1, low_s);
    setTrackbarPos("Low S","HSV Detection", low_s);
}
void on_high_s_thresh_trackbar(int, void *)
{
    high_s = max(high_s, low_s+1);
    setTrackbarPos("High S", "HSV Detection", high_s);
}
void on_low_v_thresh_trackbar(int, void *)
{
    low_v= min(high_v-1, low_v);
    setTrackbarPos("Low V","HSV Detection", low_v);
}
void on_high_v_thresh_trackbar(int, void *)
{
    high_v = max(high_v, low_v+1);
    setTrackbarPos("High V", "HSV Detection", high_v);
}
