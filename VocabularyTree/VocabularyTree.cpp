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
	root->featureNums = nFeatures;
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
	int ccount =0 ;
	curNode->children = new vocabularyTreeNode*[branchNum];
	int offset = 0;
	for(int i = 0; i < nBranch; i++) {
		curNode->children[i] = new vocabularyTreeNode(branchNum, featureLength, clusterCenter[i], nums[i], curDepth);
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
	curNode->tf = 0.0;
	for(int i = 0; i < curNode->nBranch; i++) {
		clearTF(curNode->children[i], curDepth + 1);
	}
}

void vocabularyTree::getTFIDF(vocabularyTreeNode* curNode, int curDepth, double sum, int imageID) {
	if(curDepth == depth)
		return;
	
	double tfidfValue = curNode->tf * curNode->idf;
	if(tfidfValue != 0) {
		tfidfValue /= sum;
		curNode->invertedIndex.push_back(index(imageID, tfidfValue));
	}
	for(int i = 0; i < curNode->nBranch; i++) {
		getTFIDF(curNode->children[i], curDepth + 1, sum, imageID);
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
	queue<vocabularyTreeNode*> q;
	q.push(curNode);
	while(!q.empty()) {
		vocabularyTreeNode* cur = q.front();
		q.pop();
		cout << cur->tf << " " << cur->idf << " " << cur->featureNums << endl;
		system("pause");
		if(cur->children != NULL) {
			for(int i = 0; i < cur->nBranch; i++) 
				q.push(cur->children[i]);
		}
	}
}

double vocabularyTree::HKgetSum(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth) {
		return 0;
	} else {
		double sum = curNode->tf * curNode->idf;
		for(int i = 0; i < curNode->nBranch; i++) {
			sum += HKgetSum(curNode->children[i], curDepth + 1);
		}
		return sum;
	} 
}

void vocabularyTree::HKCalDis(vocabularyTreeNode* curNode, int curDepth, vector<matchInfo>& imageDis, double sum) {
	if(curDepth == depth)
		return;
	double tfidfValue = curNode->tf * curNode->idf;
	tfidfValue /= max(sum, 1e-3);
	if(tfidfValue != 0) {
		for(int i = 0; i < curNode->invertedIndex.size(); i++) {
			imageDis[(curNode->invertedIndex)[i].imageID].dis -= 2 * tfidfValue;
		}
	}
	for(int i = 0; i < curNode->nBranch; i++)
		HKCalDis(curNode->children[i], curDepth + 1, imageDis, sum);
}