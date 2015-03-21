#include "VocabularyTree.h"

int main()
{
	imageRetriver retriver;
	retriver.buildDataBase("F:\data");
	string queryPath;
	cin >> queryPath;
	retriver.queryImage(queryPath.c_str());
	return 0;
}