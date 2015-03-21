#include "VocabularyTree.h"

//==========================functions in class imageRetriver========================
void imageRetriver::buildDataBase(char* directoryPath)
{
	vector<string> imagePaths;
	DirectoryList(directoryPath, imagePaths, ".jpg");

	double*** trainFeatures;
	int nFeatures = getTrainFeatures(trainFeatures, imagePaths);
	tree->buildTree(trainFeatures, nFeatures, BRANCHNUM, DEPTHNUM, FEATLENGTH);
	vector<vector<double>> tfidfVector = getTFIDFVector(trainFeatures, nFeatures);
	addFeature2DataBase(tfidfVector);
}

vector<string> imageRetriver::queryImage(const char* imagePath)
{
	vector<string> ans;
	return ans;
}

int imageRetriver::getTrainFeatures(double*** trainFeatures, vector<string> imagePaths)
{
	int nImages = imagePaths.size();
	trainFeatures = new double**[nImages];
	nFeatures = new int[nImages];
	int totalCounts = 0;

	for(int i = 0; i < nImages; i++)
	{
		IplImage* img = cvLoadImage(imagePaths[i].c_str());
		struct feature* feat = NULL;
		int n = sift_features(img, &feat);
		trainFeatures[i] = new double*[n];
		for(int j = 0; j < n; j++)
			trainFeatures[i][j] = feat[j].descr;
		cvReleaseImage(&img);
		totalCounts += n;
		nFeatures[i] = n;
	}

	return totalCounts;
}

vector<vector<double>> imageRetriver::getTFIDFVector(double*** features, int nFeatures)
{
	vector<vector<double>> tfidfVector;
	for(int i = 0; i < nFeatures; i++)
	{
		vector<double> oneVector = getOneTFIDFVector(features[i], nFeatures[i]);
		tfidfVector.push_back(oneVector);
	}
	return tfidfVector;
}

vector<double> imageRetriver::getOneTFIDFVector(double** oneImageFeat, int featNum)
{
	vector<double> tfidfVector;

	return tfidfVector;
}

void imageRetriver::addFeature2DataBase(vector<vector<double>> tfidfVector)
{

}