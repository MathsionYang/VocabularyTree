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

#define MAX_ITER 100000
#define THRESHOLD 0.01   //not sure

double sqr_distance(unsigned char* vector1, unsigned char* vector2, int featureLength)
{
	double sum = 0;
	for(int i = 0; i < featureLength; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[1] - vector2[i]);

	return sum;
}

void node_add(unsigned char* &vector1, unsigned char* &vector2, int featureLength)
{
	for(int i = 0; i < featureLength; i++)
	{
		vector1[i] += vector2[i];
	}
}

void node_divide_cnt(unsigned char* &vector1, int cnt, int featureLength)
{
	if(cnt == 0)
		cnt = 1e-3;

	for(int i = 0; i < featureLength; i++)
		vector1[i] /= cnt;
}

void kmeans(featureClustering* features, int nFeatures, int branchNum, int* nums, int featureLength, unsigned char** clusterCenter)
{
	nums = new int[branchNum];
	for(int i = 0; i < branchNum; i++)
		nums[i] = 0;

	if(nFeatures < branchNum)
	{
		clusterCenter = new unsigned char*[nFeatures];
		for(int i = 0; i < nFeatures; i++)
		{
			clusterCenter[i] = features[i].feature;
		}
		return;
	}

	int* idx = new int[nFeatures];
	int* cnt = new int[branchNum];

	clusterCenter = new unsigned char*[branchNum];
	for(int i = 0; i < branchNum; i++)
		clusterCenter[i] = features[i].feature;

	unsigned char** tempCenters;
	tempCenters = new unsigned char*[branchNum];
	for(int i = 0; i < branchNum; i++)
	{
		tempCenters[i] = new unsigned char[featureLength];
		for(int j = 0; j < featureLength; j++)
			tempCenters[i][j] = 0;
	}

	for(int iter = 0; iter < MAX_ITER; iter++)
	{
		memset(cnt, 0, sizeof(int) * branchNum);
		for(int i = 0; i < branchNum; i++)
			memset(tempCenters, 0, sizeof(unsigned char) * featureLength);

		for(int i = 0; i < featureLength; i++)
		{
			double mindis = 1e20;
			int minIndex = 0;
			for(int j = 0; j < branchNum; j++)
			{
				double dis = sqr_distance(clusterCenter[i], features[i].feature, featureLength);
				if(dis < mindis)
				{
					mindis = dis;
					minIndex = j;
				}
			}
			cnt[minIndex]++;
			node_add(tempCenters[idx[i] = minIndex], features[i].feature, featureLength);
		}

		for(int i = 0; i < branchNum; i++)
			node_divide_cnt(tempCenters[i], cnt[i], featureLength);

		double sum = 0;
		for(int i = 0; i < branchNum; i++)
			sum += sqr_distance(tempCenters[i], clusterCenter[i], featureLength);
		clusterCenter = tempCenters;

		if(sum < THRESHOLD || iter == MAX_ITER)
		{
			for(int i = 0; i < nFeatures; i++)
				nums[i] = cnt[i];
			break;
		}
	}

	delete[] idx;
	delete[] cnt;
	return;
}

int cmp(const void* a, const void* b)
{

}

#endif