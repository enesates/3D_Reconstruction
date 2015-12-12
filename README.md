# 3D_Reconstruction
Real time 3D reconstruction using OpenNI, OpenCV, and OpenGL.

Building OpenCV with OpenNI and OpenGL:
http://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
http://docs.opencv.org/2.4/doc/user_guide/ug_kinect.html
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_OPENNI=ON -D WITH_OPENGL=ON -D

Compiling the code:
g++ -o pointCloud pointCloud.cpp $(pkg-config opencv --cflags --libs) -lGL -lGLU -lGLEW -lglut -lglfw

