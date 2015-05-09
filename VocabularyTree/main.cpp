#include "VocabularyTree.h"

#ifdef EXPERIMENT
SYSTEMTIME timeRecord;
fstream timeFile;
#endif

int main() {
#ifdef EXPERIMENT
	timeFile.open("timeRecord.txt", ios::out | ios::app);
	timeFile << endl << endl << endl;
#endif

	imageRetriver retriver;
	retriver.buildDataBase("D:\\data\\image.orig");

#ifndef EXPERIMENT
	string queryPath;
	cout << "type in the query image path:" << endl;
	cin >> queryPath;
	vector<string> ans = retriver.queryImage(queryPath.c_str());
	printAns(ans);
#endif

#ifdef EXPERIMENT 
	FILE* fp = fopen("resultRecord.txt", "w");
	
#ifdef EXPERIMENT
	GetLocalTime(&timeRecord);
	timeFile << "start query " << timeRecord.wHour << " " << timeRecord.wMinute << " " << timeRecord.wSecond << " " << timeRecord.wMilliseconds << endl;
#endif

	for(int i = 0; i < 100; i++) {
		vector<string> ans = retriver.queryImage(retriver.databaseImagePath[i].c_str());
		fprintf(fp, "query image: %s\n", retriver.databaseImagePath[i].c_str());
		for(int j = 0; j < ans.size(); j++)
			fprintf(fp, "%s\n", ans[j].c_str());
		fprintf(fp, "\n\n\n\n");
		printf("%d ", i);
	}

#ifdef EXPERIMENT
	GetLocalTime(&timeRecord);
	timeFile << "end query " << timeRecord.wHour << " " << timeRecord.wMinute << " " << timeRecord.wSecond << " " << timeRecord.wMilliseconds << endl;
#endif

	fclose(fp);
	timeFile.close();
#endif

	system("pause");
	return 0;
}