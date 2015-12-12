#include <stdio.h>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann/flann.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

enum { CAP1 = 49, CAP2 = 50, SHOW = 51, LISTEN = -1 };

bool quit = false;

int main( int argc, char** argv )
{
    VideoCapture capture;

    Mat image_1;
    Mat image_2;
    Mat image_3;
    Mat depthMap;

    int key = LISTEN;

    capture.open( CV_CAP_OPENNI_ASUS );
    if (!capture.isOpened()){
	printf("--(!)Error opening video capture\n");
	return -1;
    }

    while(true)
    {
        if(!capture.grab()){
            printf("Can not grab images.");
            return -1;
        }

        if(key == LISTEN){
            capture.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP);
            const float scaleFactor = 0.05f;
            depthMap.convertTo(image_3, CV_8UC1, scaleFactor );

            if(image_3.empty() ){
                printf(" --(!) No captured frame -- Break!");
                break;
            }

	    imshow("Capturing", image_3);
        }

        if( key == CAP1){
            capture.retrieve(image_1, CAP_OPENNI_GRAY_IMAGE);

            if(image_1.empty()){
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            key = LISTEN;
        }

        if (key == CAP2){
            capture.retrieve(image_2, CAP_OPENNI_GRAY_IMAGE);

            if(image_2.empty()){
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            key = SHOW;
        }

        if (key == SHOW) {
            /* http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html#feature-flann-matcher
	        */
            int minHessian = 400;

            // SurfFeatureDetector detector( minHessian );
            Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian);

            std::vector<KeyPoint> keypoints_1, keypoints_2;

            //-- Step 2: Calculate descriptors (feature vectors)
            //  SurfDescriptorExtractor extractor;

            Mat descriptors_1, descriptors_2;

            surf->detectAndCompute(image_1, Mat(), keypoints_1, descriptors_1);
            surf->detectAndCompute(image_2, Mat(), keypoints_2, descriptors_2);

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
		        if(matches[i].distance <= max(2*min_dist, 0.02)){
		            good_matches.push_back( matches[i]);
		        }
            }

            //-- Draw only "good" matches
            Mat img_matches;
            drawMatches( image_1, keypoints_1, image_2, keypoints_2,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

            //-- Show detected matches
            imshow( "Good Matches", img_matches );
        }

        key = waitKey(30);
        if (key == 27)
            return 0;
    }

    return 0;
}
