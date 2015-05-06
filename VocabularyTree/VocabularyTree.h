#ifndef VOCABULARYTREE
#define VOCABULARYTREE

#include <opencv.hpp>
#include <stdio.h>
#include <iostream>
#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <windows.h>
#include <map>
#include <vector>
#include <math.h>
#include <queue>
#include <algorithm>
#include <functional>
#include <fstream>

using namespace std;

#define INTMAX 2147483647
#define INTMIN -2147483648
#define NRESULT 20
#define ANSNUM 20     //the most similiar 20 images
#define MAXFEATNUM 4000
#define DEFAULTBRANCH 10
#define DEFAULTDEPTH 6
#define DEFAULTFEATURELENGH 128

//#define BUILDTREE
#define BUILDDATABASE
#define EXPERIMENT
//#define EXTRACTFEAT
#define READFILE

typedef struct index {
	int imageID;
	double tfidf;
	index(int imageIDInput, double tfidfInput) {
		imageID = imageIDInput; 
		tfidf = tfidfInput;
	}
}index;

typedef struct matchInfo {
	double dis;
	string imagePath;
	matchInfo(double inputDis, string inputImagePath) {
		dis = inputDis;
		imagePath = inputImagePath;
	} 
}matchInfo;

class featureClustering {
public:
	double* feature;
	int label;
	featureClustering() { feature = NULL; label = 0; }
	featureClustering(double* inputFeature, int inputLabel) {feature = inputFeature; label = inputLabel;}
};

class metaData {
public:
	string metaDataPath;
	
	metaData() { metaDataPath = "metaData.dat"; }
	vector<featureClustering*> readFeature() {
		vector<featureClustering*> ans;
		double* feature = new double[DEFAULTFEATURELENGH];
		FILE* fp = fopen(metaDataPath.c_str(), "r");
		for(int i = 0; i < DEFAULTBRANCH; i++) {
			memset(feature, 0, sizeof(double) * DEFAULTFEATURELENGH);
			for(int j = 0; j < DEFAULTFEATURELENGH; j++)
				fscanf(fp, "%lf", &feature[j]);
			featureClustering* temp = new featureClustering(feature, i);
			ans.push_back(temp);
		}
		fclose(fp);
		return ans;
	}

	void writeFeature(vector<featureClustering*> feat) {
		FILE* fp = fopen(metaDataPath.c_str(), "w");
		for(int i = 0 ; i < feat.size(); i++) {
			for(int j = 0; j < DEFAULTFEATURELENGH; j++)
				fprintf(fp, "%lf ", feat[i]->feature[j]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	vector<string> readFilePath() {
		vector<string> ans;
		string temp;
		fstream file;
		file.open(metaDataPath.c_str(), ios::in);
		int count = 0;
		while(getline(file, temp)) {
			count++;
			if(count == 10)
				break;
		}
		for(int i = 0; i < DEFAULTBRANCH; i++) {
			string temp;
			file >> temp;
			ans.push_back(temp);
		}
		file.close();
		return ans;
	}

	void writeFilePath(vector<string> path) {
		fstream file;
		file.open(metaDataPath.c_str(), ios::in | ios::app);
		for(int i = 0; i < path.size(); i++) {
			file << path[i] << endl;	
		}
		file.close();
	}
};

class featureFile {
public:
	int clusterIndex;
	int featureNum;
	string filePath;
	featureFile() {clusterIndex = 0; featureNum = 0;}
	featureFile(int inputIndex, string inputPath) { clusterIndex = inputIndex; filePath = inputPath; }

	void writeClusterCenter(int clusterIndex) {
		char fileName[200];
		sprintf(fileName, "%d.dat", clusterIndex);
		filePath = fileName;
		fstream file;
		file.open(filePath.c_str(), ios::out);
		file << clusterIndex << endl;
		file.close();
	}

	void writeOneFeat(double* feature, int featureLength, int clusterIndex) {
		FILE* file = fopen(filePath.c_str(), "a+");
		for(int i = 0; i < featureLength; i++)
			fprintf(file, "%lf ", feature[i]);
		fprintf(file, "\n");
		fclose(file);
	}

	void readFeatures(double**& feature, int featureLength, int& featureNum) {
		FILE* file = fopen(filePath.c_str(), "r");	
		char lines[100000];
		while(fgets(lines, 100000, file))
			featureNum++;
		fclose(file);
		
		featureNum -= 1;
		file = fopen(filePath.c_str(), "r");
		int clusterIndex = 0;
		fscanf(file, "%d", &clusterIndex);
		feature = new double*[featureNum];
		for(int i = 0; i < featureNum; i++)
			feature[i] = new double[featureLength];
		for(int i = 0; i < featureNum; i++) {
			for(int j = 0; j < featureLength; j++) {
				fscanf(file, "%lf", &feature[i][j]);
			}
		}
		fclose(file);
	}
};

class vocabularyTreeNode {
public:
	int nBranch;
	int featureLength;
	double* feature;
	double weight;
	vocabularyTreeNode** children;
	int featureNums;
	int depth;

	vocabularyTreeNode() {nBranch = DEFAULTBRANCH; featureLength = DEFAULTFEATURELENGH; children = NULL; depth = 0;}
	vocabularyTreeNode(int branchNum, int inputFeatureLength, double* features, int featureNumber, int curDepth = 0) {
		nBranch = branchNum;
		featureLength = inputFeatureLength;
		feature = features;
		featureNums = featureNumber;
		children = NULL;
		depth = curDepth;
	}

	double tf;
	double idf;
	bool add;                     //the tf varible can be added per image once

	vector<index> invertedIndex;
};

class vocabularyTree {
public:
	vocabularyTreeNode* root;
	int nNodes;
	int nBranch;
	int depth;

	vocabularyTree() { root = new vocabularyTreeNode(); nBranch = DEFAULTBRANCH; depth = DEFAULTDEPTH; }
	void buildTree(vector<featureClustering*> originCenter, featureFile* fileRecord, int nBranch, int depth, int featureLength);
	void buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength);
	void clearTF(vocabularyTreeNode* root, int curDepth);
	void getTFIDF(vocabularyTreeNode* curNode, int curDepth, double sum, int imageID);
	void clearADD(vocabularyTreeNode* root, int curDepth);
	void printTree(vocabularyTreeNode* root, int curDepth);
	double HKgetSum(vocabularyTreeNode* root, int curDepth);
	void HKCalDis(vocabularyTreeNode* curNode, int curDepth, vector<matchInfo>& imageDis, double sum);
};

class imageRetriver {
public:
	vocabularyTree* tree;
	vector<string> databaseImagePath; 
	int featureLength;  
	int nImages;
	int totalFeatures;
	featureFile* featFile;
	metaData MetaData;

	imageRetriver() { tree = new vocabularyTree(); nImages = 0; featureLength = DEFAULTFEATURELENGH;}
	void buildDataBase( char* directoryPath );
	vector<string> queryImage( const char* imagePath ); 

	void getOriginCenter(vector<featureClustering*>& originCenter, queue<feature*>& featRecord);
	int getTrainFeatures(vector<featureClustering*> originCenter, queue<feature*>& featRecord);
	void calIDF(double** features, int nFeatures);               //cal IDF for each node in the tree
	void getTFIDFVector(featureFile* featFile, int nImages);
	void getOneTFIDFVector(double** features, int featNums, int nStart, int imageCount); 
	vector<string> calImageDis(double** queryFeat, vector<matchInfo> &imageDis, int nFeatures);

	void HKAdd(double* feature, int depth, vocabularyTreeNode* node, bool checkAdd);
	void HKDiv(vocabularyTreeNode* curNode, int curDepth);
};

extern double sqr_distance(double* vector1, double* vector2, int featureLength);
extern double vector_sqr_distance(vector<double> vector1, vector<double> vector2);
extern void vector_normalize(vector<double>& vector1);
extern void node_add(double* &vector1, double* &vector2, int featureLength);
extern void node_divide_cnt(double* &vector1, int cnt, int featureLength);
extern void kmeans(featureClustering* &features, int nFeatures, int branchNum, int* &nums, int featureLength, double** &clusterCenter);
extern int cmp(const void* a, const void* b);
extern bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext);
extern void printAns(vector<string> ans);
extern void saveImageFeature(int imageID, double** feature, int nFeatures);
extern void readImageFeature(int imageID, double** &feature, int& nFeatures);

#endif