#ifndef VOCABULARYTREE
#define VOCABULARYTREE

#include <opencv.hpp>
#include <stdio.h>
#include <iostream>
#include "sift.h"
#include <stdlib.h>

using namespace std;

#define INTMAX 2147483647
#define INTMIN -2147483648
#define NRESULT 20

class vocabularyTreeNode
{
public:
	int nBranch;
	int nFeatures;
	unsigned char* feature;
	double weight;
	vocabularyTreeNode** children;

	vocabularyTreeNode() {}
	vocabularyTreeNode(int branchNum, int featureLength, unsigned char* features)
	{
		nBranch = branchNum;
		nFeatures = featureLength;
		feature = features;
	}
};

class featureClustering
{
public:
	unsigned char* feature;
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
	void buildTree(unsigned char** features, int nFeatures, int nBranch, int depth, int featureLength);
	void buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength);
};

class imageRetriver
{
public:
	vocabularyTree* tree;
	float** imageFeatures;            //database
	int nImages;
	int featureLength;

	imageRetriver() { tree = NULL; imageFeatures = NULL; nImages = 0; featureLength = 0; }
	void buildDataBase( char* directoryPath );
	int* queryImage( char* imagePath ); 
};

extern double sqr_distance(unsigned char* vector1, unsigned char* vector2, int featureLength);
extern void node_add(unsigned char* &vector1, unsigned char* &vector2, int featureLength);
extern void node_divide_cnt(unsigned char* &vector1, int cnt, int featureLength);
extern void kmeans(featureClustering* features, int nFeatures, int branchNum, int* nums, int featureLength, unsigned char** clusterCenter);
extern int cmp(const void* a, const void* b);

#endif