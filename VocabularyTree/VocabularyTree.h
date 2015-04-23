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

using namespace std;

#define INTMAX 2147483647
#define INTMIN -2147483648
#define NRESULT 20
#define ANSNUM 20     //the most similiar 20 images
#define MAXFEATNUM 10000
#define DEFAULTBRANCH 10
#define DEFAULTDEPTH 6
#define DEFAULTFEATURELENGH 128

//#define BUILDTREE
#define BUILDDATABASE
//#define EXPERIMENT

class vocabularyTreeNode {
public:
	int nBranch;
	int featureLength;
	double* feature;
	double weight;
	vocabularyTreeNode** children;
	int featureNums;

	vocabularyTreeNode() {nBranch = DEFAULTBRANCH; featureLength = DEFAULTFEATURELENGH;}
	vocabularyTreeNode(int branchNum, int inputFeatureLength, double* features, int featureNumber) {
		nBranch = branchNum;
		featureLength = inputFeatureLength;
		feature = features;
		featureNums = featureNumber;
	}

	double tf;
	double idf;
	bool add;                     //the tf varible can be added per image once
};

class featureClustering {
public:
	double* feature;
	int label;
	featureClustering() { feature = NULL; label = 0; }
};

class vocabularyTree {
public:
	vocabularyTreeNode* root;
	int nNodes;
	int nBranch;
	int depth;

	vocabularyTree() { root = new vocabularyTreeNode(); nBranch = DEFAULTBRANCH; depth = DEFAULTDEPTH; }
	void buildTree(double** features, int nFeatures, int nBranch, int depth, int featureLength);
	void buildRecursion(int curDepth, vocabularyTreeNode* curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength);
	void clearTF(vocabularyTreeNode* root, int curDepth);
	void getTFIDF(vector<double>& tfidf, vocabularyTreeNode* curNode, int curDepth);
	void clearADD(vocabularyTreeNode* root, int curDepth);
	void printTree(vocabularyTreeNode* root, int curDepth);
};

class imageRetriver {
public:
	vocabularyTree* tree;
	map<vector<double>, string> imageDatabase;
	vector<string> databaseImagePath; 
	int featureLength;  
	int nImages;
	int *nFeatures;                        //features per image
	int totalFeatures;

	imageRetriver() { tree = new vocabularyTree(); nImages = 0; featureLength = DEFAULTFEATURELENGH; nFeatures = NULL;}
	void buildDataBase( char* directoryPath );
	vector<string> queryImage( const char* imagePath ); 

	int getTrainFeatures(double** &features, vector<string> imagePaths);
	void calIDF(double** features);               //cal IDF for each node in the tree
	vector<vector<double>> getTFIDFVector(double** features, int nImages);
	vector<double> getOneTFIDFVector(double** features, int featNums, int nStart); 
	void addFeature2DataBase(vector<vector<double>> tfidfVector);

	void HKAdd(double* feature, int depth, vocabularyTreeNode* node);
	void HKDiv(vocabularyTreeNode* curNode, int curDepth);
};

extern double sqr_distance(double* vector1, double* vector2, int featureLength);
extern double vector_sqr_distance(vector<double> vector1, vector<double> vector2);
extern void node_add(double* &vector1, double* &vector2, int featureLength);
extern void node_divide_cnt(double* &vector1, int cnt, int featureLength);
extern void kmeans(featureClustering* &features, int nFeatures, int branchNum, int* &nums, int featureLength, double** &clusterCenter);
extern int cmp(const void* a, const void* b);
extern bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext);
extern void printAns(vector<string> ans);

#endif