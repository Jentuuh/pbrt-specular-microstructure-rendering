#include "Image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

Image::Image(std::string filename)
{
	int normalWidth;
	int normalHeight;
	int numChannels;

	unsigned char* normalMap = stbi_load("../Data/normal3.png", &normalWidth, &normalHeight, &numChannels, 3);
	this->pixels = std::vector<char>(normalMap, normalMap + normalWidth * normalHeight * numChannels);
	this->width = normalWidth;
	this->height = normalHeight;
	this->numChannels = numChannels;

	std::cout << "Loaded image of dimensions: (" << normalWidth << "," << normalHeight << "," << numChannels  << ")" << std::endl;
}


Image::Image(int width, int height)
{
	this->pixels = std::vector<char>(width * height * 3, 0.0f);
	this->width = width;
	this->height = height;
	this->numChannels = 3;
}


char* Image::getPixel(int x, int y)
{
	char pixel[3];

	pixel[0] = pixels[(x * width + y) * this->numChannels];
	pixel[1] = pixels[(x * width + y) * this->numChannels + 1];
	pixel[2] = pixels[(x * width + y) * this->numChannels + 2];

	return pixel;
}

void Image::writePixel(int x, int y, char rgb[3])
{
	this->pixels[(x * width + y) * this->numChannels] = rgb[0];
	this->pixels[(x * width + y) * this->numChannels + 1] = rgb[1];
	this->pixels[(x * width + y) * this->numChannels + 2] = rgb[2];
}


void Image::saveImage(std::string fileName)
{
	stbi_write_png(fileName.c_str(), this->width, this->height, this->numChannels, this->pixels.data(), this->width * this->numChannels * sizeof(char));
}


