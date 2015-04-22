#include "VocabularyTree.h"

//==========================functions in class imageRetriver========================
void imageRetriver::buildDataBase(char* directoryPath) {
	vector<string> databaseImagePath;
	DirectoryList(directoryPath, databaseImagePath, ".jpg");
	double** trainFeatures = NULL;
	printf("extract features...\n");
	int nFeatures = getTrainFeatures(trainFeatures, databaseImagePath);

	printf("build tree...\n");
	tree->buildTree(trainFeatures, nFeatures, tree->nBranch, tree->depth, featureLength);
	printf("build tree finished...\n");
	
	printf("build database...\n");
	vector<vector<double>> tfidfVector = getTFIDFVector(trainFeatures, nFeatures);
	addFeature2DataBase(tfidfVector);
}

vector<string> imageRetriver::queryImage(const char* imagePath) {
	vector<string> ans;
	IplImage* img = cvLoadImage(imagePath);
	struct feature* feat = NULL;
	int nFeatures = sift_features(img, &feat);
	double** queryFeat = new double*[nFeatures];
	for(int i = 0; i < nFeatures; i++)
		queryFeat[i] = feat[i].descr;

	vector<double> tfidfVector = getOneTFIDFVector(queryFeat, nFeatures, 0);

	multimap<double, string> candidates;
	map<vector<double>, string>::iterator iter; 
	double maxDistance = 1e20;
	for(iter = imageDatabase.begin(); iter != imageDatabase.end(); iter++) {
		vector<double> cur = iter->first;
		double distance = vector_sqr_distance(cur, tfidfVector);
		candidates.insert(make_pair(distance, iter->second));
	}

	int count = 0;
	multimap<double, string>::iterator iter1;
	for(iter1 = candidates.begin(); iter1 != candidates.end(); iter1++) {
		ans.push_back(iter1->second);
		count++;
		if(count == ANSNUM)
			break;
	}

	return ans;
}

int imageRetriver::getTrainFeatures(double** &trainFeatures, vector<string> imagePaths) {
	nImages = imagePaths.size();

#ifndef EXPERIMENT              //less images for faster speed in debug
	nImages /= 100;
	printf("total images %d\n", nImages);
#endif

	trainFeatures = new double*[nImages * MAXFEATNUM];
	nFeatures = new int[nImages];
	int featCount = 0;
	for(int i = 0; i < nImages; i++) {
		cout << imagePaths[i] << endl;
		IplImage* img = cvLoadImage(imagePaths[i].c_str());
		struct feature* feat = NULL;
		int n = sift_features(img, &feat);
		for(int j = 0; j < n; j++) {
			trainFeatures[featCount] = feat[j].descr;
			featCount++;
		}
		cvReleaseImage(&img);
		nFeatures[i] = n;
	}

	return featCount;
}

void imageRetriver::HKAdd(double* feature, int depth, vocabularyTreeNode* cur) {  //add tf value for each node
	if(depth == tree->depth)
		return;
	if(cur->add) {
		cur->tf++;
		cur->add = false;
	}
	int minIndex = 0;
	double minDis = 1e20;
	for(int i = 0; i < cur->nBranch; i++) {
		if((cur->children[i])->featureNums == 0)
			continue;
		double curDis = sqr_distance(feature, (cur->children[i])->feature, featureLength);
		if(curDis < minDis) {
			minDis = curDis;
			minIndex = i;
		}
	}
	HKAdd(feature, depth + 1, cur->children[minIndex]);
}

void imageRetriver::HKDiv(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == tree->depth)
		return;
	curNode->idf = 1.0 * nImages / (max(curNode->tf, 1));
	for(int i = 0; i < curNode->nBranch; i++) {
		HKDiv(curNode->children[i], curDepth + 1); 
	}
}

void imageRetriver::calIDF(double** features) {
	int featureCount = 0;
	for(int i = 0; i < nImages; i++) {
		tree->clearADD(tree->root, 0);
		for(int j = 0; j < nFeatures[i]; j++) {
			HKAdd(features[featureCount], 0, tree->root);   //add the number of images at least one descriptor path through for each node
			featureCount++;
		}
	}
	HKDiv(tree->root, 0);    //cal N / Ni, where N is the number of total images and Ni is tf
}

vector<vector<double>> imageRetriver::getTFIDFVector(double** features, int nImages) { 
	calIDF(features);        //calculate idf for each node in the tree
	tree->printTree(tree->root, 0);
	vector<vector<double>> tfidfVector;
	int featureCount = 0;
	for(int i = 0; i < nImages; i++) {
		vector<double> oneImgTFIDF = getOneTFIDFVector(features, nFeatures[i], featureCount);
		tfidfVector.push_back(oneImgTFIDF);
		featureCount += nFeatures[i];
	}
	return tfidfVector;
}

vector<double> imageRetriver::getOneTFIDFVector(double** features, int featNums, int nStart) {
	tree->clearTF(tree->root, 0);
	for(int i = 0; i < featNums; i++) 
		HKAdd(features[nStart + i], 0, tree->root);
	vector<double> oneImgTFIDF;
	tree->getTFIDF(oneImgTFIDF, tree->root, 0);
	return oneImgTFIDF;
}

void imageRetriver::addFeature2DataBase(vector<vector<double>> tfidfVector) {
	for(int i = 0; i < nImages; i++) {
		imageDatabase.insert(make_pair(tfidfVector[i], imagePath[i]));
	}
}