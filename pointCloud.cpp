#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glfw.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann/flann.hpp"

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

enum { CAP1 = 49, CAP2 = 50, SHOW = 51, LISTEN = -1 };

GLint   windowWidth  = 1280;     // Define our window width
GLint   windowHeight = 960;     // Define our window height
GLfloat fieldOfView  = 45.0f;   // FoV
GLfloat zNear        = 0.1f;    // Near clip plane
GLfloat zFar         = 200.0f;  // Far clip plane

// Frame counting and limiting
int    frameCount = 0;
double frameStartTime, frameEndTime, frameDrawTime;
float camera_0 = 0.0;
float camera_1 = 0.0;
float camera_2 = 0.0;

bool quit = false;
bool corCap = false;

void draw(cv::Mat &camFrame, cv::Mat &depthFrame)
{
    int w = depthFrame.rows;
    int h = depthFrame.cols;

    glViewport (0, 0, (GLsizei)windowWidth, (GLsizei)windowHeight);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (60, (GLfloat)windowWidth / (GLfloat)windowHeight, zNear, zFar);
    glMatrixMode (GL_MODELVIEW);

    // Clear the screen and depth buffer, and reset the ModelView matrix to identity
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(0.0f, 0.0f, -4.0f);
    glPointSize(0.1);

    glRotatef(camera_0, 1.0, 0.0, 0.0);
    glRotatef(camera_1, 0.0, 1.0, 0.0);
    glRotatef(camera_2, 0.0, 0.0, 1.0);

    float h2 = h/2.0;
    float w2 = w/2.0;

    glBegin(GL_POINTS);
    for (int j=0; j<w; j++){
        for (int i=0; i<h; i++) {
            glVertex3f((i-h2)/h2, (j-w2)/w2, (int) depthFrame.at<unsigned short>(j,i)/255.0);
        }
    }

    glEnd();
}

void handleKeypress(int theKey, int theAction)
{
    if (theAction == GLFW_PRESS)
    {
        switch (theKey)
        {
            case GLFW_KEY_ESC:
                quit = true;
                break;
            case GLFW_KEY_PAGEUP:
                camera_2 -= 10;
                break;
            case GLFW_KEY_PAGEDOWN:
                camera_2 += 10;
                break;
            case GLFW_KEY_UP:
                camera_0 -= 10;
                break;
            case GLFW_KEY_DOWN:
                camera_0 += 10;
                break;
            case GLFW_KEY_LEFT:
                camera_1 -= 10;
                break;
            //case GLFW_KEY_RIGHT:
              //  camera_1 += 10;
                //break;
            case GLFW_KEY_RIGHT:
                corCap = true;
                break;
            default:
                break;

        } // End of switch statement

    } // End of GLFW_PRESS
}

void initGL()
{

    /* http://r3dux.org/2012/01/how-to-convert-an-opencv-cvmat-to-an-opengl-texture/ */
    // Define our buffer settings
    int redBits    = 8, greenBits = 8,  blueBits    = 8;
    int alphaBits  = 8, depthBits = 24, stencilBits = 8;

    // Initialise glfw
    glfwInit();

    // Create a window
    if(!glfwOpenWindow(windowWidth, windowHeight, redBits, greenBits, blueBits, alphaBits, depthBits, stencilBits, GLFW_WINDOW))
    {
        cout << "Failed to open window!" << endl;
        glfwTerminate();
        exit(-1);
    }

    glfwSetWindowTitle("OpenCV/OpenNI Sensor Data to Texture | r3dux");

    // Specify the callback function for key presses/releases
    glfwSetKeyCallback(handleKeypress);

    //  Initialise glew (must occur AFTER window creation or glew will error)
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        cout << "GLEW initialisation error: " << glewGetErrorString(err) << endl;
        exit(-1);
    }
    cout << "GLEW okay - using version: " << glewGetString(GLEW_VERSION) << endl;
    // to check glew is working
    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    printf("%u\n", vertexBuffer);

    // Setup our viewport to be the entire size of the window
    glViewport(0, 0, (GLsizei)windowWidth, (GLsizei)windowHeight);

    // Change to the projection matrix and set our viewing volume
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // The following code is a fancy bit of math that is equivilant to calling:
    // gluPerspective(fieldOfView/2.0f, width/height , near, far);
    // We do it this way simply to avoid requiring glu.h
    GLfloat aspectRatio = (windowWidth > windowHeight)? float(windowWidth)/float(windowHeight) : float(windowHeight)/float(windowWidth);
    GLfloat fH = tan( float(fieldOfView / 360.0f * 3.14159f) ) * zNear;
    GLfloat fW = fH * aspectRatio;
    glFrustum(-fW, fW, -fH, fH, zNear, zFar);

    // ----- OpenGL settings -----

    glDepthFunc(GL_LEQUAL);		// Specify depth function to use
    glEnable(GL_DEPTH_TEST);    // Enable the depth buffer
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Ask for nicest perspective correction
    glEnable(GL_CULL_FACE);     // Cull back facing polygons
    glfwSwapInterval(1);        // Lock screen updates to vertical refresh

    // Switch to ModelView matrix and reset
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set our clear colour to black
}

void lockFramerate(double framerate)
{
    // Note: frameStartTime is called first thing in the main loop
    // Our allowed frame time is 1 second divided by the desired FPS
    static double allowedFrameTime = 1.0 / framerate;
    // Get current time
    frameEndTime = glfwGetTime();
    // Calc frame draw time
    frameDrawTime = frameEndTime - frameStartTime;
    double sleepTime = 0.0;

    // Sleep if we've got time to kill before the next frame
    if (frameDrawTime < allowedFrameTime)
    {
        sleepTime = allowedFrameTime - frameDrawTime;
        glfwSleep(sleepTime);
    }

    // Debug stuff
    double potentialFPS = 1.0 / frameDrawTime;
    double lockedFPS    = 1.0 / (glfwGetTime() - frameStartTime);
    // cout << "Draw: " << frameDrawTime << " Sleep: " << sleepTime;
    // cout << " Pot. FPS: " << potentialFPS << " Locked FPS: " << lockedFPS << endl;
}

int main( int argc, char** argv )
{
    initGL();

    VideoCapture capture;
    Mat camFrame;
    Mat depthFrame;
    Mat prevFrame;

    capture.open( CV_CAP_OPENNI_ASUS );
    if (! capture.isOpened()) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}

    do {
        frameStartTime = glfwGetTime();

        if( !capture.grab() ) {
            cout << "Can not grab images." << endl;
            return -1;
        }

        else {
            capture.retrieve(camFrame, CAP_OPENNI_BGR_IMAGE);
            capture.retrieve(depthFrame, CAP_OPENNI_DEPTH_MAP);
            // const float scaleFactor = 0.05f;
            // depthMap.convertTo( depthFrame, CV_8UC1, scaleFactor );
            // Draw texture contents
            draw(camFrame, depthFrame);
            // Swap the active and visual pages

            glfwSwapBuffers();
        }

	// test with corresponding points
	if (corCap == true) {

  	    if(prevFrame.empty())
		prevFrame = camFrame.clone();
	  
	    else {
		// FIX IT!!!
		int minHessian = 400;

		//SurfFeatureDetector detector( minHessian );

		Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian);
		std::vector<KeyPoint> keypoints_1, keypoints_2;
	  
		//-- Step 2: Calculate descriptors (feature vectors)
		//  SurfDescriptorExtractor extractor;

		Mat descriptors_1, descriptors_2;

		surf->detectAndCompute(prevFrame, Mat(), keypoints_1, descriptors_1);
		surf->detectAndCompute(camFrame, Mat(), keypoints_2, descriptors_2);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );

		double max_dist = 0; double min_dist = 100;
		//-- Quick calculation of max and min distances between keypoints

		for( int i = 0; i < descriptors_1.rows; i++ )
		{
		    double dist = matches[i].distance;
		    if( dist < min_dist ) min_dist = dist;
	 	    if( dist > max_dist ) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;

		for( int i = 0; i < descriptors_1.rows; i++ )
		{ 
		    if( matches[i].distance <= max(2*min_dist, 0.02) ) {
			good_matches.push_back( matches[i]);
	            }
		}

		Mat img_matches;
		// change positions: camFrame, prevFrame
		drawMatches( prevFrame, keypoints_1, camFrame, keypoints_2, good_matches,
			img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Show detected matches
		waitKey(30);
		imshow( "Good Matches", img_matches );

		for( int i = 0; i < (int)good_matches.size(); i++ )
		{
		    printf( "-- Good Match [%d] K1: %d (%f - %f)  -- K2: %d (%f - %f) \n", i, good_matches[i].queryIdx,
			keypoints_1[good_matches[i].queryIdx].pt.x,
			keypoints_1[good_matches[i].queryIdx].pt.y,
			good_matches[i].trainIdx,
			keypoints_2[good_matches[i].trainIdx].pt.x,
			keypoints_2[good_matches[i].trainIdx].pt.y );
		}

		/*  Mat K = (Mat_<double>(3,3) << 5.4656902770673435e+02, 0., 3.1912501985122083e+02, 0., 5.4483615626372955e+02, 2.4745007963107062e+02, 0., 0., 1.);
		Mat fundamentalMatrix = findFundamentalMat(Mat(points1), Mat(points2), CV_FM_RANSAC, 3, 0.99);
			Mat E = K.t() * fundamentalMatrix * K;
	
		SVD svd(E);
		Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
		Matx33d Winv(0, 1, 0, -1, 0, 0, 0, 0, 1);
	
		Mat R = svd.u * Mat(W) * svd.vt;
		Mat t = svd.u.col(2);
		P1 = Matx34d(
				R(0,0), R(0,1), R(0,2), t(0),
				R(1,0), R(1,1), R(1,2), t(1),
				R(2,0), R(2,1), R(2,2), t(2));
	
		Mat normalizedTrackedPts,normalizedBootstrapPts;
		undistortPoints(Points<float>(keypoints_1), normalizedTrackedPts, camFrame, Mat());
		undistortPoints(Points<float>(keypoints_2), normalizedBootstrapPts, camFrame, Mat());
	
		//triangulate
		Mat pt_3d_h(4,keypoints_1.size(),CV_32FC1);
		triangulatePoints(prevFrame,camFrame,normalizedBootstrapPts,normalizedTrackedPts,pt_3d_h);
		*/

		/*  FileStorage fsI("camera.yml", FileStorage::READ);
		Mat matI1, matI2, matD1, matD2;
		fsI["camera_matrix"] >> matI1;
		fsI["camera_matrix"] >> matI2;
		fsI["distortion_coefficients"] >> matD1;
		fsI["distortion_coefficients"] >> matD2;

		CvMat camInt1 = matI1, camInt2 = matI2, camDist1 = matD1, camDist2 = matD2;

		FileStorage fs("camera.yml", FileStorage::READ);
		Mat mat1, mat2;
		fs["extrinsic_parameters"] >> mat1;
		fs["extrinsic_parameters"] >> mat2;

		double pointImg1_a[2] = { 640, 480};
		Mat pointImg1 = Mat(2,1, CV_64FC1, pointImg1_a);
		CvMat _pointImg1 = cvMat(1,1,CV_64FC2,pointImg1_a);
		double pointImg2_a[2] = { 640, 460};
		Mat pointImg2 = Mat(2,1, CV_64FC1, pointImg2_a);
		CvMat _pointImg2 = cvMat(1,1,CV_64FC2,pointImg2_a);

		cvUndistortPoints(&_pointImg1,&_pointImg1,&camInt1,&camDist1);
		cvUndistortPoints(&_pointImg2,&_pointImg2,&camInt2,&camDist2);

		Mat point4D = Mat(4,1, CV_64FC1);
		cv::triangulatePoints(mat1, mat2, pointImg1, pointImg2, point4D);

		double w = point4D.at<double>(3,0);
		double x = point4D.at<double>(0,0)/w;
		double y = point4D.at<double>(1,0)/w;
		double z = point4D.at<double>(2,0)/w;

		cout << x << ", " << y << ", " << z << endl;
		*/

		prevFrame = camFrame.clone();
   	    }

	    corCap = false;
	}

	// Quit out if the OpenGL window was closed
	if (!glfwGetWindowParam( GLFW_OPENED))
		quit = true;

	frameCount++;
	// Lock our main loop to 30fps
	lockFramerate(30.0);
	// if( cv::waitKey( 30 ) >= 0 )
	// break;

    } while (quit == false);

    capture.release();
    glfwTerminate();

    return 0;
}
