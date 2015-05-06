#include "VocabularyTree.h"

//==========================functions in class imageRetriver========================
void imageRetriver::buildDataBase(char* directoryPath) {
	DirectoryList(directoryPath, databaseImagePath, ".jpg");
	vector<featureClustering*> originCenter;
#ifdef EXTRACTFEAT
	printf("extract features...\n");
	queue<feature*> featRecord;
	getOriginCenter(originCenter, featRecord);      //得到初始聚类中心
	MetaData.writeFeature(originCenter);
	getTrainFeatures(originCenter, featRecord);  //将所有特征初分类,featData返回特征文件的路径
	for(int i = 0; i < featRecord.size(); i++) {
		feature* topFeat = featRecord.front();
		free(topFeat);
		featRecord.pop();
	}
#endif
#ifdef READFILE
	nImages = databaseImagePath.size();
	originCenter = MetaData.readFeature();
	featFile = new featureFile[DEFAULTBRANCH];
	vector<string> clusterPath = MetaData.readFilePath();
	for(int i = 0; i < DEFAULTBRANCH; i++)
		featFile[i] = new featureFile(i, clusterPath[i]);
#endif	
	tree->buildTree(originCenter, featFile, tree->nBranch, tree->depth, featureLength);  //建树
	printf("build database...\n");
	getTFIDFVector(featFile, nImages);
}

void imageRetriver::getOriginCenter(vector<featureClustering*>& originCenter, queue<feature*>& featRecord) {
	int imageNum = databaseImagePath.size();
	int divNum = imageNum / DEFAULTBRANCH;
	int clusterCenter = 0;
	for(int i = 0; i < imageNum; i++) {
		if(i % divNum == 0) {
			IplImage* img = cvLoadImage(databaseImagePath[i].c_str());
			struct feature* feat = NULL;
			int nFeatures = sift_features(img, &feat);
			featRecord.push(feat);
			featureClustering* tempCenter = new featureClustering;
			tempCenter->label = clusterCenter;
			tempCenter->feature = feat[0].descr;
			originCenter.push_back(tempCenter);
			clusterCenter++;
			cvReleaseImage(&img);
		}
	}
}

int imageRetriver::getTrainFeatures(vector<featureClustering*> originCenter, queue<feature*>& featRecord) {
	nImages = databaseImagePath.size();
#ifndef EXPERIMENT              //less images for faster speed in debug
	nImages /= 100;
	printf("total images %d\n", nImages);
#endif

	featFile = new featureFile[DEFAULTBRANCH];
	vector<string> featFilePath;
	for(int i = 0; i < DEFAULTBRANCH; i++) {
		featFile[i].writeClusterCenter(i);
		featFilePath.push_back(featFile[i].filePath);
	}
	MetaData.writeFilePath(featFilePath);
	struct feature* feat = NULL;
	for(int i = 0; i < databaseImagePath.size(); i++) {
		printf("%s\n", databaseImagePath[i].c_str());
		IplImage* img = NULL;
		img = cvLoadImage(databaseImagePath[i].c_str());
		if(!img) continue;
		int nFeatures = sift_features(img, &feat);
		cvReleaseImage(&img);
		featRecord.push(feat);
		totalFeatures += nFeatures;
		for(int j = 0; j < nFeatures; j++) {
			double minDis = 1e20;
			int minIndex = 0;
			for(int k = 0; k < DEFAULTBRANCH; k++) {
				double tempDis = sqr_distance(feat[j].descr, originCenter[k]->feature, featureLength);
				if(tempDis < minDis) {
					minDis = tempDis;
					minIndex = k;
				}
			}
			featFile[minIndex].writeOneFeat(feat[j].descr, featureLength, minIndex);
			featFile[minIndex].featureNum++;
		}
		printf("%lf ", feat[0].descr[0]);

		double** features = new double*[nFeatures];
		for(int j = 0; j < nFeatures; j++) {
			features[j] = feat[j].descr;
		}
		saveImageFeature(i, features, nFeatures);
		delete[] features;
	}
	tree->root->featureNums = totalFeatures;
	printf("total features %d\n", totalFeatures);
	return 0;
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


void imageRetriver::getTFIDFVector(featureFile* featFile, int nImages) {
	for(int i = 0; i < nImages; i++) {
		double** features = NULL;
		int nFeatures = 0;
		readImageFeature(i, features, nFeatures);
		calIDF(features, nFeatures);
	}
	HKDiv(tree->root, 0); 

	for(int i = 0; i < nImages; i++) {
		double** features = NULL;
		int nFeatures = 0;
		readImageFeature(i, features, nFeatures);
		getOneTFIDFVector(features, nFeatures, 0, i);
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

void imageRetriver::calIDF(double** features, int nFeatures) {
	int featureCount = 0;
	tree->clearADD(tree->root, 0);
	for(int i = 0; i < nFeatures; i++) {
		HKAdd(features[featureCount], 0, tree->root, true);   //add the number of images at least one descriptor path through for each node
		featureCount++;
	}
	//HKDiv(tree->root, 0);    //cal N / Ni, where N is the number of total images and Ni is tf
}