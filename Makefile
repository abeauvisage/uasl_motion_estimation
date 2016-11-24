CC = g++
BUILD = Debug

CFLAGS = -Wall -fexceptions -std=c++11 -g -fPIC
LFLAGS = -L/usr/local/lib
LIBS =  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
INCLUDES = -Iinclude -I/usr/include/local -I../MotionEstimation/include
TARGET = bin/$(BUILD)/libMotionEstimation
SOURCES = src/Graph2D.cpp src/viso_stereo.cpp

OBJS = $(SOURCES:.cpp=.o)

all : $(TARGET)

$(TARGET) : main.cpp $(OBJS)
	$(CC) $(INCLUDES) $(LFLAGS) $(LIBS) -o $(TARGET) $(OBJS) main.cpp 

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
