#include "VocabularyTree.h"

void vocabularyTree::buildTree(unsigned char** features, int nFeatures, int nBranch, int depth, int featureLength)
{
	featureClustering* feature2Cluster;
	feature2Cluster = new featureClustering[nFeatures];
	for(int i = 0; i < nFeatures; i++)
	{
		feature2Cluster[i].label = 0;
		feature2Cluster[i].feature = features[i];
	}

	buildRecursion(0, root, feature2Cluster, nFeatures, nBranch, featureLength);
}

void vocabularyTree::buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength)
{
	if(curDepth == depth)
		return;
	
	int* nums;
	nums = new int[branchNum];
	unsigned char** clusterCenter = NULL;

	kmeans(features, nFeatures, branchNum, nums, featureLength, clusterCenter);
	qsort(features, nFeatures, sizeof(featureClustering*), cmp);

	curNode->children = new vocabularyTreeNode*[branchNum];
	int offset = 0;
	for(int i = 0; i < nBranch; i++)
	{
		curNode->children[i] = new vocabularyTreeNode(branchNum, featureLength, clusterCenter[i]);
		buildRecursion(curDepth + 1, curNode->children[i], features + offset, nums[i], branchNum, featureLength);
		offset += nums[i] * sizeof(featureClustering);
	}
}

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
			features[i].label = minIndex;
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
	return ((featureClustering*)a)->label - ((featureClustering*)b)->label;
}
