#include "VocabularyTree.h"

int main() {
/*
	imageRetriver retriver;
	retriver.buildDataBase("F:\data");
	string queryPath;
	cin >> queryPath;
	retriver.queryImage(queryPath.c_str());
*/
	IplImage* img = cvLoadImage("fate_1.jpg");
	cvNamedWindow("test", 1);
	cvShowImage("test", img);
	cvWaitKey(0);
	cvDestroyWindow("test");
	cvReleaseImage(&img);
	return 0;
}