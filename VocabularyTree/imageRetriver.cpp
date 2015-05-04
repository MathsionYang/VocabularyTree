#include "VocabularyTree.h"

//==========================functions in class imageRetriver========================
void imageRetriver::buildDataBase(char* directoryPath) {
	DirectoryList(directoryPath, databaseImagePath, ".jpg");
	double** trainFeatures = NULL;
	printf("extract features...\n");
	struct feature* feat = NULL;
	queue<feature*> featRecord;
	int nFeatures = getTrainFeatures(trainFeatures, databaseImagePath, feat, featRecord);

	printf("build tree...\n");
	tree->buildTree(trainFeatures, nFeatures, tree->nBranch, tree->depth, featureLength);
	printf("build tree finished...\n");
	
	printf("build database...\n");
	getTFIDFVector(trainFeatures, nImages);
	for(int i = 0; i < featRecord.size(); i++) {
		feature* topFeat = featRecord.front();
		free(topFeat);
		featRecord.pop();
	}
}

vector<string> imageRetriver::queryImage(const char* imagePath) {
	IplImage* img = cvLoadImage(imagePath);
	struct feature* feat = NULL;
	int nFeatures = sift_features(img, &feat);
	double** queryFeat = new double*[nFeatures];
	for(int i = 0; i < nFeatures; i++)
		queryFeat[i] = feat[i].descr;

	vector<matchInfo> imageDis;
	for(int i = 0; i < nImages; i++) {
		imageDis.push_back(matchInfo(2, databaseImagePath[i]));
	}
	vector<string> ans = calImageDis(queryFeat, imageDis, nFeatures);
	return ans;
}

int imageRetriver::getTrainFeatures(double** &trainFeatures, vector<string> imagePaths, feature* &feat, queue<feature*>& featRecord) {
	nImages = imagePaths.size();

#ifndef EXPERIMENT              //less images for faster speed in debug
	nImages /= 100;
	printf("total images %d\n", nImages);
#endif

	trainFeatures = new double*[nImages * MAXFEATNUM];
	nFeatures = new int[nImages];
	int featCount = 0;
	for(int i = 0; i < nImages; i++) {
		cout << imagePaths[i] << " ";
		databaseImagePath.push_back(imagePaths[i]);
		IplImage* img = cvLoadImage(imagePaths[i].c_str());
		int n = sift_features(img, &feat);
		featRecord.push(feat);
		for(int j = 0; j < n; j++) {
			trainFeatures[featCount] = feat[j].descr;
			featCount++;
		}
		cout << n << endl;
		cvReleaseImage(&img);
		nFeatures[i] = n;
	}
	cout << "total features: " << featCount << endl;
	return featCount;
}

void imageRetriver::HKAdd(double* feature, int depth, vocabularyTreeNode* cur, bool checkAdd) {  //add tf value for each node
	if(depth == tree->depth) {
		return;
	}
	if(cur->add | !checkAdd) {
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
	HKAdd(feature, depth + 1, cur->children[minIndex], checkAdd);
}

void imageRetriver::HKDiv(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == tree->depth)
		return;
	curNode->idf = log(1.0 * nImages / (max((int)curNode->tf, 1)));
	for(int i = 0; i < curNode->nBranch; i++) {
		HKDiv(curNode->children[i], curDepth + 1); 
	}
}

void imageRetriver::calIDF(double** features) {
	int featureCount = 0;
	for(int i = 0; i < nImages; i++) {
		tree->clearADD(tree->root, 0);
		for(int j = 0; j < nFeatures[i]; j++) {
			HKAdd(features[featureCount], 0, tree->root, true);   //add the number of images at least one descriptor path through for each node
			featureCount++;
		}
	}
	HKDiv(tree->root, 0);    //cal N / Ni, where N is the number of total images and Ni is tf
}

void imageRetriver::getTFIDFVector(double** features, int nImages) { 
	calIDF(features);        //calculate idf for each node in the tree
	//tree->printTree(tree->root, 0);
	vector<vector<double>> tfidfVector;
	int featureCount = 0;
	for(int i = 0; i < nImages; i++) {
		getOneTFIDFVector(features, nFeatures[i], featureCount, i);
		featureCount += nFeatures[i];
	}
}

void imageRetriver::getOneTFIDFVector(double** features, int featNums, int nStart, int imageID) {
#ifdef BUILDDATABASE
	printf("image %d nStart %d featnums %d\n", imageID, nStart, featNums);
#endif
	tree->clearTF(tree->root, 0);
	for(int i = 0; i < featNums; i++) 
		HKAdd(features[nStart + i], 0, tree->root, false);
	double sum = tree->HKgetSum(tree->root, 0);
	tree->getTFIDF(tree->root, 0, sum, imageID);
}

void imageRetriver::addFeature2DataBase(vector<vector<double>> tfidfVector) {
#ifdef BUILDDATABASE
	printf("add feature to database\n");
#endif
	for(int i = 0; i < nImages; i++) {
#ifdef BUILDDATABASE
		printf("%s\n", databaseImagePath[i].c_str());
#endif
		imageDatabase.insert(make_pair(tfidfVector[i], databaseImagePath[i]));
	}
}

bool matchInfoCmp(matchInfo& a, matchInfo& b) {
	return a.dis < b.dis;
}

vector<string> imageRetriver::calImageDis(double** queryFeat, vector<matchInfo> &imageDis, int nFeatures) {
	tree->clearTF(tree->root, 0);
	for(int i = 0; i < nFeatures; i++) {
		HKAdd(queryFeat[i], 0, tree->root, false);
	}
	double sum = tree->HKgetSum(tree->root, 0);
	tree->HKCalDis(tree->root, 0, imageDis, sum);
	sort(imageDis.begin(), imageDis.end(), matchInfoCmp);
	vector<string> ans;
	for(int i = 0; i < ANSNUM; i++) {
		ans.push_back(imageDis[i].imagePath);
		cout << imageDis[i].dis << " ";
	}

	return ans;
}