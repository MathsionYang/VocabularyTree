#include "VocabularyTree.h"

//==========================other functions===================================
#define MAX_ITER 100000
#define ENDTHRESHOLD 0.0001   //not sure

double sqr_distance(double* vector1, double* vector2, int featureLength) {
	double sum = 0;
	for(int i = 0; i < featureLength; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
	
	return sum;
}

double vector_sqr_distance(vector<double> vector1, vector<double> vector2) {
	int size = vector1.size();
	double sum = 0;
	for(int i = 0; i < size; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);

	return sum;
}

void vector_normalize(vector<double> &vector1) {
	double sum = 0;
	int length = vector1.size();
	for(int i = 0; i < length; i++) {
		sum += vector1[i];
	}
	sum = max(sum, 1e-3);
	for(int i = 0; i < length; i++)
		vector1[i] = vector1[i] / sum;
}
	 
void node_add(double* &vector1, double* &vector2, int featureLength) {
	for(int i = 0; i < featureLength; i++) {
		vector1[i] += vector2[i];
	}
}

void node_divide_cnt(double* &vector1, int cnt, int featureLength) {
	if(cnt == 0)
		cnt = 1;

	for(int i = 0; i < featureLength; i++) {
		vector1[i] /= cnt;
	}
}


void kmeans(featureClustering* &features, int nFeatures, int branchNum, int* &nums, int featureLength, double** &clusterCenter) {
	nums = new int[branchNum];
	for(int i = 0; i < branchNum; i++)
		nums[i] = 0;

	if(nFeatures < branchNum) {
		clusterCenter = new double*[branchNum];
		for(int i = 0; i < branchNum; i++) {
			clusterCenter[i] = new double[featureLength];
			memset(clusterCenter[i], 0, sizeof(double) * featureLength);
		}
		for(int i = 0; i < nFeatures; i++) {
			memcpy(clusterCenter[i], features[i].feature, sizeof(double) * featureLength);
			features[i].label = i;
			nums[i] = 1;
		}
#ifdef BUILDTREE
		cout << "features is less than num of branches" << endl;
#endif
		return;
	}

	int* cnt = new int[branchNum];

	clusterCenter = new double*[branchNum];
	for(int i = 0; i < branchNum; i++)
		clusterCenter[i] = new double[featureLength];
	for(int i = 0; i < branchNum; i++) {
		memcpy(clusterCenter[i], features[i].feature, sizeof(double) * featureLength);
	}
	
	double** tempCenters;
	tempCenters = new double*[branchNum];
	for(int i = 0; i < branchNum; i++) {
		tempCenters[i] = new double[featureLength];
		memset(tempCenters[i], 0, sizeof(double) * featureLength);
	}

	for(int iter = 0; iter < MAX_ITER; iter++) {
		memset(cnt, 0, sizeof(int) * branchNum);
		for(int i = 0; i < branchNum; i++) {
			memset(tempCenters[i], 0, sizeof(double) * featureLength);
		}
		
		for(int i = 0; i < nFeatures; i++) {
			double mindis = 1e10;
			int minIndex = 0;
			for(int j = 0; j < branchNum; j++) {
				double dis = sqr_distance(clusterCenter[j], features[i].feature, featureLength);
				if(dis < mindis) {
					mindis = dis;
					minIndex = j;
				}
			}
			cnt[minIndex]++;
			features[i].label = minIndex;
			node_add(tempCenters[minIndex], features[i].feature, featureLength);
		}

#ifdef BUILDTREE
		printf("iter %d times, nodes in each branch: ", iter);
		for(int i = 0; i < branchNum; i++)
			printf("%d ", cnt[i]);
		printf("\n");
#endif

		for(int i = 0; i < branchNum; i++)
			node_divide_cnt(tempCenters[i], cnt[i], featureLength);

		double sum = 0;
		for(int i = 0; i < branchNum; i++)
			sum += sqr_distance(tempCenters[i], clusterCenter[i], featureLength);
#ifdef BUILDTREE
		cout << "error: " << sum << endl;
		
#endif
		for(int i = 0; i < branchNum; i++)
			for(int j = 0; j < featureLength; j++)
				clusterCenter[i][j] = tempCenters[i][j]; 

		if(sum < ENDTHRESHOLD || iter == MAX_ITER) {
			for(int i = 0; i < branchNum; i++)
				nums[i] = cnt[i];
			break;
		}
	}

	delete[] cnt;
	
	return;
}

int cmp(const void* a, const void* b) {
	return ((featureClustering*)a)->label - ((featureClustering*)b)->label;
}

#define LEN 1024
bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext) {
	WIN32_FIND_DATA FindData;
	HANDLE hError;
	int FileCount = 0;
	char FilePathName[LEN];
	char FullPathName[LEN];
	strcpy(FilePathName, Path);
	strcat(FilePathName, "\\*.*");
	hError = FindFirstFile(FilePathName, &FindData);
	if (hError == INVALID_HANDLE_VALUE) {
		printf("error");
		return 0;
	}
	while(::FindNextFile(hError, &FindData)) {
		if (strcmp(FindData.cFileName, ".") == 0 
		 || strcmp(FindData.cFileName, "..") == 0 ) {
			continue;
		}
  
		wsprintf(FullPathName, "%s\\%s", Path,FindData.cFileName);
		FileCount++;
		string temp = FullPathName;
		if(temp.find(ext) != temp.npos) 
			path.push_back(string(temp));

		if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			printf("<Dir>");
			DirectoryList(FullPathName, path, ext);
		}
	}
}

void printAns(vector<string> ans) {
	for(int i = 0; i < ans.size(); i++)
		printf("%s\n", ans[i].c_str());
}

void saveImageFeature(int imageID, double** feature, int nFeatures) {
	char name[100];
	sprintf(name, "features/%d.dat", imageID);
	FILE* fp = fopen(name, "w");
	fprintf(fp, "%d\n", nFeatures);
	for(int i = 0; i < nFeatures; i++) {
		for(int j = 0; j < DEFAULTFEATURELENGH; j++)
			fprintf(fp, "%lf ", feature[i][j]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void readImageFeature(int imageID, double** &feature, int& nFeatures) {
	char name[100];
	sprintf(name, "features/%d.dat", imageID);
	FILE* fp = fopen(name, "r");
	fscanf(fp, "%d", &nFeatures);
	feature = new double*[nFeatures];
	for(int i = 0; i < nFeatures; i++)
		feature[i] = new double[DEFAULTFEATURELENGH];
	for(int i = 0; i < nFeatures; i++) {
		for(int j = 0; j < DEFAULTFEATURELENGH; j++) {
			fscanf(fp, "%lf", &feature[i][j]);
		}
	}
	fclose(fp);
}

vocabularyTree* readTree() {
	vocabularyTree* tree = new vocabularyTree();
	FILE* fp = fopen("tree.dat", "r");
	fscanf(fp, "%d %d %d\n", &tree->nNodes, &tree->nBranch, &tree->depth);
	queue<vocabularyTreeNode*> q;
	q.push(tree->root);
	while(!q.empty()) {
		vocabularyTreeNode* cur = q.front();
		q.pop();
		if(cur->depth == DEFAULTDEPTH) {
			continue;
		} else {
			fscanf(fp, "%d %d %lf %d %d %lf %lf %d\n", &cur->nBranch, &cur->featureLength, &cur->weight, &cur->featureNums, &cur->depth, &cur->tf, &cur->idf, &cur->add);
			cur->feature = new double[DEFAULTFEATURELENGH];
			if(cur->depth != 0) {
				for(int i = 0; i < DEFAULTFEATURELENGH; i++)
					fscanf(fp, "%lf ", &cur->feature[i]);
			}
			int invertedIndexSize = 0;
			fscanf(fp, "%d", &invertedIndexSize);
			for(int i = 0; i < invertedIndexSize; i++) {
				int inputID = 0;
				double inputInverted = 0;
				fscanf(fp, "%d %lf", &inputID, &inputInverted);
				cur->invertedIndex.push_back(index(inputID, inputInverted));
			}
			if(cur->depth != 5) {
				cur->children = new vocabularyTreeNode*[DEFAULTBRANCH];
				for(int i = 0; i < DEFAULTBRANCH; i++) {
					cur->children[i] = new vocabularyTreeNode();
					q.push(cur->children[i]);
				}
			}
		}
	}
	fclose(fp);
	printf("finish\n");
	return tree;
}

void writeTree(vocabularyTree* tree) {
	queue<vocabularyTreeNode*> q;
	q.push(tree->root);
	FILE* fp = fopen("tree.dat", "w");
	fprintf(fp, "%d %d %d\n", tree->nNodes, tree->nBranch, tree->depth);
	while(!q.empty()) {
		vocabularyTreeNode* cur = q.front();
		q.pop();
		fprintf(fp, "%d %d %lf %d %d %lf %lf %d\n", cur->nBranch, cur->featureLength, cur->weight, cur->featureNums, cur->depth, cur->tf, cur->idf, cur->add);
		if(cur->depth != 0) {
			for(int i = 0; i < DEFAULTFEATURELENGH; i++) {
				fprintf(fp, "%lf ", cur->feature[i]);
			}
		}
		fprintf(fp, "%d ", cur->invertedIndex.size());
		for(int i = 0; i < cur->invertedIndex.size(); i++) {
			fprintf(fp, "%d %lf ", cur->invertedIndex[i].imageID, cur->invertedIndex[i].tfidf);
		}
		fprintf(fp, "\n");
		if(cur->children != NULL) {
			for(int i = 0; i < cur->nBranch; i++)
				q.push(cur->children[i]);
		}
	}
	fclose(fp);
	printf("finish write\n");
}
