#pragma once

#include <string>
#include <vector>

class Image
{
public:
	Image(std::string filename, int imageChannels); // Image from file
	Image(int width, int height); // Empty image

	int* getPixel(int x, int y, int* pixel);
	void writePixel(int x, int y, char rgb[3]);

	void saveImage(std::string fileName);

	int width;
	int height;
private:
	std::vector<char> pixels;
	int numChannels;
};

