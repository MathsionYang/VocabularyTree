#include "VocabularyTree.h"

//==========================other functions===================================
#define MAX_ITER 100000
#define ENDTHRESHOLD 0.01   //not sure

double sqr_distance(double* vector1, double* vector2, int featureLength) {
	double sum = 0;
	for(int i = 0; i < featureLength; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[1] - vector2[i]);

	return sum;
}

double vector_sqr_distance(vector<double> vector1, vector<double> vector2) {
	int size = vector1.size();
	double sum = 0;
	for(int i = 0; i < size; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);

	return sum;
}

void node_add(double* &vector1, double* &vector2, int featureLength) {
	for(int i = 0; i < featureLength; i++) {
		vector1[i] += vector2[i];
	}
}

void node_divide_cnt(double* &vector1, int cnt, int featureLength) {
	if(cnt == 0)
		cnt = 1e-3;

	for(int i = 0; i < featureLength; i++)
		vector1[i] /= cnt;
}

void kmeans(featureClustering* features, int nFeatures, int branchNum, int* nums, int featureLength, double** clusterCenter) {
	nums = new int[branchNum];
	for(int i = 0; i < branchNum; i++)
		nums[i] = 0;

	if(nFeatures < branchNum) {
		clusterCenter = new double*[nFeatures];
		for(int i = 0; i < nFeatures; i++) {
			clusterCenter[i] = features[i].feature;
		}
		return;
	}

	int* idx = new int[nFeatures];
	int* cnt = new int[branchNum];

	clusterCenter = new double*[branchNum];
	for(int i = 0; i < branchNum; i++)
		clusterCenter[i] = features[i].feature;

	double** tempCenters;
	tempCenters = new double*[branchNum];
	for(int i = 0; i < branchNum; i++) {
		tempCenters[i] = new double[featureLength];
		memset(tempCenters[i], 0, sizeof(double) * featureLength);
	}

	for(int iter = 0; iter < MAX_ITER; iter++) {
		memset(cnt, 0, sizeof(int) * branchNum);
		for(int i = 0; i < branchNum; i++)
			memset(tempCenters[i], 0, sizeof(double) * featureLength);

		for(int i = 0; i < nFeatures; i++) {
			double mindis = 1e20;
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
			idx[i] = minIndex;
		}
#ifdef DEBUG
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
		clusterCenter = tempCenters;
#ifdef DEBUG
		printf("error: %lf\n", sum);
#endif
		if(sum < ENDTHRESHOLD || iter == MAX_ITER) {
			for(int i = 0; i < branchNum; i++)
				nums[i] = cnt[i];
			break;
		}
	}

	delete[] idx;
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