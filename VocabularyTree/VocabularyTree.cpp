#include "VocabularyTree.h"

//==========================functions in class vocabularyTree========================
void vocabularyTree::buildTree(double** features, int nFeatures, int nBranch, int depth, int featureLength) {
#ifdef BUILDTREE
	printf("nFeatures %d\n", nFeatures);
#endif

	featureClustering* feature2Cluster;
	feature2Cluster = new featureClustering[nFeatures];
	for(int i = 0; i < nFeatures; i++) {
		feature2Cluster[i].label = 0;
		feature2Cluster[i].feature = features[i];
	}
	buildRecursion(0, root, feature2Cluster, nFeatures, nBranch, featureLength);
}

void vocabularyTree::buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength) {
	if(curDepth == depth)
		return;

#ifdef BUILDTREE
	printf("depth %d\n", curDepth);
#endif

	int* nums;
	nums = new int[branchNum];
	double** clusterCenter = NULL;
	kmeans(features, nFeatures, branchNum, nums, featureLength, clusterCenter);
	qsort(features, nFeatures, sizeof(featureClustering), cmp);
	curNode->children = new vocabularyTreeNode*[branchNum];
	int offset = 0;
	for(int i = 0; i < nBranch; i++) {
		curNode->children[i] = new vocabularyTreeNode(branchNum, featureLength, clusterCenter[i], nums[i]);

#ifdef BUILDTREE
		printf("build new node, nums of features:%d depth %d\n", nums[i], curDepth);
		//system("pause");
#endif

		buildRecursion(curDepth + 1, curNode->children[i], features + offset, nums[i], branchNum, featureLength);
		offset += nums[i];
	}
}

void vocabularyTree::clearTF(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;
	curNode->tf = 0;
	for(int i = 0; i < curNode->nBranch; i++) {
		clearTF(curNode->children[i], curDepth + 1);
	}
}

void vocabularyTree::getTFIDF(vector<double>& tfidf, vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;

	tfidf.push_back(curNode->tf * curNode->idf);
	for(int i = 0; i < curNode->nBranch; i++) {
		getTFIDF(tfidf, curNode->children[i], curDepth + 1);
	}
}

void vocabularyTree::clearADD(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;
	curNode->add = true;
	for(int i = 0; i < curNode->nBranch; i++)
		clearADD(curNode->children[i], curDepth + 1);
}

void vocabularyTree::printTree(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;
	printf("%lf ", curNode->idf);
	system("pause");
	for(int i = 0; i < curNode->nBranch; i++)
		printTree(curNode->children[i], curDepth + 1);
}