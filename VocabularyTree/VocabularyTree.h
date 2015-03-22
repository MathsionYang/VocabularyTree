#ifndef VOCABULARYTREE
#define VOCABULARYTREE

#include <opencv.hpp>
#include <stdio.h>
#include <iostream>
#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <windows.h>
#include <map>
#include <vector>

using namespace std;

#define INTMAX 2147483647
#define INTMIN -2147483648
#define NRESULT 20
#define BRANCHNUM 10
#define DEPTHNUM 6
#define FEATLENGTH 128
#define ANSNUM 20     //the most similiar 20 images
#define MAXFEATNUM 1500

class vocabularyTreeNode
{
public:
	int nBranch;
	int nFeatures;
	double* feature;
	double weight;
	vocabularyTreeNode** children;

	vocabularyTreeNode() {}
	vocabularyTreeNode(int branchNum, int featureLength, double* features)
	{
		nBranch = branchNum;
		nFeatures = featureLength;
		feature = features;
	}

	double tf;
	double idf;
};

class featureClustering
{
public:
	double* feature;
	int label;
	featureClustering() { feature = NULL; label = 0; }
};

class vocabularyTree
{
public:
	vocabularyTreeNode* root;
	int nNodes;
	int nBranch;
	int depth;

	vocabularyTree() { root = NULL; }
	void buildTree(double** features, int nFeatures, int nBranch, int depth, int featureLength);
	void buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength);
};

class imageRetriver
{
public:
	vocabularyTree* tree;
	map<vector<vector<double>>, vector<string>> imageDatabase;
	vector<string> imagePath;
	int nImages;
	int featureLength;
	int *nFeatures;
	int totalFeatures;

	imageRetriver() { tree = NULL; nImages = 0; featureLength = 0; nFeatures = NULL;}
	void buildDataBase( char* directoryPath );
	vector<string> queryImage( const char* imagePath ); 

	int getTrainFeatures(double** features, vector<string> imagePaths);
	vector<vector<double>> getTFIDFVector(double** features, int nImages);
	vector<double> getOneTFIDFVector(double* oneImageFeat, int featNums, int nStart); 
	void addFeature2DataBase(vector<vector<double>> tfidfVector);
};

extern double sqr_distance(double* vector1, double* vector2, int featureLength);
extern void node_add(double* &vector1, double* &vector2, int featureLength);
extern void node_divide_cnt(double* &vector1, int cnt, int featureLength);
extern void kmeans(featureClustering* features, int nFeatures, int branchNum, int* nums, int featureLength, double** clusterCenter);
extern int cmp(const void* a, const void* b);
extern bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext);

#endif