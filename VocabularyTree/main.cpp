#include "VocabularyTree.h"

int main() {
	imageRetriver retriver;
	retriver.buildDataBase("F:\\data\\images");
	string queryPath;
	cin >> queryPath;
	retriver.queryImage(queryPath.c_str());
}