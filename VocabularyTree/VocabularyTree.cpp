#include "VocabularyTree.h"

//==========================functions in class vocabularyTree========================
void vocabularyTree::buildTree(vector<featureClustering*> originCenter, featureFile* fileRecord, int nBranch, int depth, int featureLength) {
	root->children = new vocabularyTreeNode*[nBranch];
	for(int i = 0; i < originCenter.size(); i++) {
		printf("build subtree %d\n", i);
		root->children[i] = new vocabularyTreeNode(nBranch, featureLength, originCenter[i]->feature, 0, 1);  //nFeatures先写0，读进来以后再补	
		double** features = NULL;
		int featNum = 0;
		fileRecord[i].readFeatures(features, featureLength, featNum);
		root->children[i]->featureNums = featNum;
		featureClustering* featureCluster = new featureClustering[featNum];
		for(int j = 0; j < featNum; j++) {
			featureCluster[j].feature = features[j];
			featureCluster[j].label = i;
		}
		buildRecursion(2, root->children[i], featureCluster, featNum, nBranch, featureLength);

		for(int j = 0; j < featNum; j++) {
			delete features[j];
		}
		delete[] featureCluster;
		delete[] features;
	}
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
	int ccount = 0;
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
	if(curNode->children != NULL) {
		for(int i = 0; i < curNode->nBranch; i++) {
			clearTF(curNode->children[i], curDepth + 1);
		}
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
	if(curNode->children != NULL) {
		for(int i = 0; i < curNode->nBranch; i++) {
			getTFIDF(curNode->children[i], curDepth + 1, sum, imageID);
		}	
	}
}

void vocabularyTree::clearADD(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;
	curNode->add = true;
	if(curNode->children != NULL) {
		for(int i = 0; i < curNode->nBranch; i++) {
			clearADD(curNode->children[i], curDepth + 1);
		}
	}
}

void vocabularyTree::printTree(vocabularyTreeNode* curNode, int curDepth) {
	queue<vocabularyTreeNode*> q;
	q.push(curNode);
	while(!q.empty()) {
		vocabularyTreeNode* cur = q.front();
		q.pop();
		cout << cur->tf << " " << cur->idf << " " << cur->featureNums << " " << cur->depth << " " << cur->feature[0]<< endl;
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
		if(curNode->children != NULL) {
			for(int i = 0; i < curNode->nBranch; i++) {
				sum += HKgetSum(curNode->children[i], curDepth + 1);
			}
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
	if(curNode->children != NULL) {
		for(int i = 0; i < curNode->nBranch; i++)
			HKCalDis(curNode->children[i], curDepth + 1, imageDis, sum);
	}
}