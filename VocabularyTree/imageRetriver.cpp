#include "VocabularyTree.h"

struct cmpClass {
	bool cmp(const double a, const double b) {
		return (a > b);
	}
};

//==========================functions in class imageRetriver========================
void imageRetriver::buildDataBase(char* directoryPath) {
	vector<string> imagePaths;
	DirectoryList(directoryPath, imagePaths, ".jpg");

	double** trainFeatures = NULL;
	int nFeatures = getTrainFeatures(trainFeatures, imagePaths);
	tree->buildTree(trainFeatures, nFeatures, tree->nBranch, tree->depth, featureLength);
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

	vector<double> tfidfVector = getOneTFIDFVector(queryFeat[0], nFeatures, 0);

	multimap<double, string, cmpClass> candidates;
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

int imageRetriver::getTrainFeatures(double** trainFeatures, vector<string> imagePaths) {
	int nImages = imagePaths.size();
	trainFeatures = new double*[nImages * MAXFEATNUM];
	nFeatures = new int[nImages];
	int featCount = 0;

	for(int i = 0; i < nImages; i++) {
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
	calIDF();
	return featCount;
}

void imageRetriver::HKAdd(double* feature, int depth, vocabularyTreeNode* cur) {
	if(depth == tree->depth)
		return;
	if(cur->add)
		cur->tf++;
	int minIndex = 0;
	double minDis = 1e20;
	for(int i = 0; i < cur->nBranch; i++) {
		double curDis = sqr_distance(feature, (cur->children[i])->feature);
		if(curDis < minDis) {
			curDis = minDis;
			minIndex = i;
		}
	}
	HKAdd(feature, depth + 1, cur->children[minIndex]);
}

void imageRetriver::calIDF(double** features) {
	int featureCount = 0;
	for(int i = 0; i < nImages; i++) {
		tree->clearTF();
		for(int j = 0; j < nFeatures[i]; j++) {
			HKAdd(features[featureCount], 0, tree->root);
			featureCount++;
		}
	}
}

vector<vector<double>> imageRetriver::getTFIDFVector(double** features, int nImages) { 
	calIDF();
	vector<vector<double>> tfidfVector;
	int startNum = 0;
	for(int i = 0; i < nImages; i++) {
		vector<double> oneVector = getOneTFIDFVector(features[i], nFeatures[i], startNum, 0);
		tfidfVector.push_back(oneVector);
		startNum += nFeatures[i];
	}
	return tfidfVector;
}

vector<double> imageRetriver::getOneTFIDFVector(double* oneImageFeat, int featNum, int startNum, int depth) {
	vector<double> tfidfVector;

	return tfidfVector;
}

void imageRetriver::addFeature2DataBase(vector<vector<double>> tfidfVector) {

}

