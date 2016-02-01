/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#define PTHREAD

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// Thread for camera "server" ensures that we always have the newest camera data
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef PTHREAD
#include <pthread.h>
#endif






#include "serial_port.h"

using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection for drone localization";
const char* keys  =
        "{d        |              | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{ci       | 0            | Camera id if input doesnt come from video (-v) }"
        "{c        |              | Camera intrinsic parameters. Needed for marker pose }"
        "{cp       |              | Camera position and rotation file. Default is camra at origin pointing up, y oppoite of N, x opposite of E }"
        "{dp       |              | File of marker detector parameters }"
        "{r        |              | show rejected candidates too }"
        "{dv       | /dev/ttyUSB0 | Device for Mavlink }"
        "{db       | 57600        | Baudrate for Mavlink device }";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    std::cout << "CamMatrix is: " << camMatrix << std::endl;
    std::cout << "DistCoeffs is: " << distCoeffs << std::endl;
    return true;
}

/**
 */
static bool readCameraPosParameters(string filename, Eigen::Vector3d &C_p_NED, Eigen::Quaternion<double> &NED_R_C) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    Vec4d tmp_R;
    Vec3d tmp_p;

    fs["camera_pos"] >> tmp_p;
    fs["camera_rot"] >> tmp_R;

    std::cout << "Camera position in NED is: " << tmp_p << std::endl;
    std::cout << "Camera orientation quaternion is: " << tmp_R << std::endl;
    NED_R_C.w() = tmp_R(0);
    NED_R_C.x() = tmp_R(1);
    NED_R_C.y() = tmp_R(2);
    NED_R_C.z() = tmp_R(3);
    C_p_NED(0) = tmp_p(0);
    C_p_NED(1) = tmp_p(1);
    C_p_NED(2) = tmp_p(2);

    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params->doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}


/**
*/
static void 
sendPosition( const Eigen::Vector3d &t, const Eigen::Vector3d &r, Serial_Port &serial_port,
const Eigen::Vector3d &C_p_NED, const Eigen::Quaternion<double> &NED_R_C )
{
    // Init structures
    mavlink_message_t message;
    mavlink_vision_position_estimate_t vision_position_estimate;
    mavlink_att_pos_mocap_t att_pos_mocap;

    // Calculate rotation quaternion. No need to pull in Eigen or other lib. for a few equations
    Eigen::Quaternion<double> C_R_f; C_R_f = Eigen::AngleAxis<double>( r.norm(), r/r.norm() );

    // Calculate position
    Eigen::Vector3d f_p_C = t;

    // Transform to NED coordinate system
    Eigen::Quaternion<double> q = NED_R_C*C_R_f;
    Eigen::Vector3d p = C_p_NED + NED_R_C._transformVector(f_p_C);

    // Parse mocap data
    att_pos_mocap.q[0] = q.w(); // Storage order in Eigen is x, y, z, w. Constructor is w, x, y, z though.
    att_pos_mocap.q[1] = q.x();
    att_pos_mocap.q[2] = q.y();
    att_pos_mocap.q[3] = q.z();
    att_pos_mocap.x = p(0);
    att_pos_mocap.y = p(1);
    att_pos_mocap.z = p(2);

    // Parse location data (this is acually not used, but is passed to QGroundControl)
    vision_position_estimate.x = p(0);
    vision_position_estimate.y = p(1);
    vision_position_estimate.z = p(2);
    q = q.conjugate();
    vision_position_estimate.roll = atan2(2*q.w()*q.x()-2*q.y()*q.z() , 1 - 2*q.x()*q.x() - 2*q.y()*q.y());
    vision_position_estimate.pitch = asin(2*q.w()*q.y() - 2*q.z()*q.x());
    vision_position_estimate.yaw = atan2(2*q.w()*q.z()-2*q.x()*q.y(), 1 - 2*q.y()*q.y() - 2*q.z()*q.z());
    q = q.conjugate();

    // Send over radio to PX4
    unsigned len = mavlink_msg_vision_position_estimate_encode(1, 1, &message, &vision_position_estimate);
    serial_port.write_message(message);
    len = mavlink_msg_att_pos_mocap_encode(1, 1, &message, &att_pos_mocap);
    serial_port.write_message(message);
}



static volatile bool exitAllThreads = false;
Mat inputImage; 
mutex m_newImage;
condition_variable cv_newImage;

void cameraThread(int camId)
{
    #ifdef PTHREAD
    pthread_t thId = pthread_self();
    pthread_attr_t thAttr;
    int policy = 0;
    int max_prio_for_policy = 0;

    pthread_attr_init(&thAttr);
    pthread_attr_getschedpolicy(&thAttr, &policy);
    max_prio_for_policy = sched_get_priority_max(policy);


    pthread_setschedprio(thId, max_prio_for_policy);
    pthread_attr_destroy(&thAttr);
    #endif

    VideoCapture inputVideo;
    inputVideo.open(camId);
    inputVideo.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT,1080);


    int totalIterations = 0;
    double tick = (double)getTickCount();

    while (!exitAllThreads) {
        inputVideo.grab();
        inputVideo.grab();
        inputVideo.retrieve(inputImage);
        cv_newImage.notify_one();

        totalIterations++;
        double dTick = (double)getTickCount();
        double currentTime = ((double)dTick - tick) / getTickFrequency();
        tick = dTick;
        if(totalIterations % 30 == 0) {
            cout << "Grabbing Time = " << currentTime * 1000 << " ms "
                 << "Frame Rate = " << 1.0/currentTime << " FPS " << endl;
        }

    }

}


/**
 */
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    bool hasCameraPos = parser.has("cp");

    // Default camera pose
    Eigen::Quaternion<double> NED_R_C; NED_R_C = Eigen::AngleAxis<double>( M_PI, Eigen::Vector3d( -sqrt(2.)/2., sqrt(2.)/2., 0 ) );
    Eigen::Vector3d C_p_NED(0, 0, 0);
    if(hasCameraPos) {
        bool readOk = readCameraPosParameters(parser.get<string>("cp"), C_p_NED, NED_R_C);
        if(!readOk) {
            cerr << "Invalid camera position file" << endl;
            return 0;
        }
    }

    Ptr<aruco::DetectorParameters> detectorParams(new aruco::DetectorParameters());
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->doCornerRefinement = true; // do corner refinement in markers

    int camId = parser.get<int>("ci");

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    } else {
        cout << "WARNING! No camera calibration file provided. Drone will not be tracked!" << endl;
    }

    /*
     * Open serial port for passing messages to pixhawk
     */
    string uart_name = "CommandLineParser should set this to default!";
    if(parser.has("dv")) {
    	uart_name = parser.get<string>("dv");
    }
    int baudrate = 57600;
    if(parser.has("db")) {
    	baudrate = parser.get<int>("db");
    }
    char * tmp = (char*)uart_name.c_str();
    Serial_Port serial_port(tmp, baudrate);
    serial_port.start();

    /*
     * Open window for displaying tracking
     */
    namedWindow("out", WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    resizeWindow("out", 1920/2, 1080/2);

    /*
     * Inintalize video device
     */
    int waitTime = 1;
    // TODO: start thread
    thread camera_thread(cameraThread,camId);

    double totalTime = 0;
    int totalIterations = 0;

    /*
     * Create Aruco board. TODO: oad this from file
     */
    std::vector<std::vector<cv::Point3f> > boardPoints(1,std::vector<cv::Point3f>(4));
    std::vector<int> boardIds(1);

    boardPoints[0][0] = Point3f( -0.05, 0.05, 0.12 );
    boardPoints[0][1] = Point3f( 0.05, 0.05, 0.12 );
    boardPoints[0][2] = Point3f( 0.05, -0.05, 0.12 );
    boardPoints[0][3] = Point3f( -0.05, -0.05, 0.12 );
    boardIds[0] = 0;

    Ptr<aruco::Board> board(new aruco::Board());
    board->objPoints = boardPoints;
    board->dictionary = dictionary;
    board->ids = boardIds;

    while(true) {
        Mat imageCopy;
        // wait for a new image
        {
            unique_lock<mutex> loopLock(m_newImage);
            cv_newImage.wait(loopLock);
        }  
        Mat image = inputImage.clone();
        

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        //vector< Vec3d > rvecs, tvecs;
        Vec3d rvecs, tvecs;

        // detect markers and estimate pose
        int boardDetected = 0;
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(estimatePose && ids.size() > 0) {
            boardDetected = aruco::estimatePoseBoard( corners, ids, board, camMatrix, distCoeffs, rvecs,
                                             tvecs);

            if( boardDetected ) {
                if(totalIterations % 10 == 0)
                cout << "Markers at:" << tvecs[0] << endl;

                // Send position to PX4
                sendPosition( Eigen::Vector3d( tvecs(0), tvecs(1), tvecs(2) ),
                    Eigen::Vector3d( rvecs(0), rvecs(1), rvecs(2) ),
                    serial_port, C_p_NED, NED_R_C );
            }  

        }
        

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 10 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if(estimatePose && boardDetected) {
                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs, tvecs,
                                    0.05f);
            }
        }

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    exitAllThreads = true;
    camera_thread.join();

    return 0;
}

