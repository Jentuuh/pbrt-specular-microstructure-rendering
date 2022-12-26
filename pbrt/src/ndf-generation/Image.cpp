#include "Image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

Image::Image(std::string filename, int imageChannels)
{
	int normalWidth;
	int normalHeight;
	int numChannels;

	unsigned char* normalMap = stbi_load(filename.c_str(), &normalWidth, &normalHeight, &numChannels, imageChannels);
	this->pixels = std::vector<char>(normalMap, normalMap + normalWidth * normalHeight * numChannels);
	this->width = normalWidth;
	this->height = normalHeight;
	this->numChannels = numChannels;

	//std::cout << "Loaded image of dimensions: (" << normalWidth << "," << normalHeight << "," << numChannels  << ")" << std::endl;
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

	pixel[0] = pixels[(y * width + x) * this->numChannels];
	pixel[1] = pixels[(y * width + x) * this->numChannels + 1];
	pixel[2] = pixels[(y * width + x) * this->numChannels + 2];

	return pixel;
}

void Image::writePixel(int x, int y, char rgb[3])
{
	this->pixels[(y * width + x) * this->numChannels] = rgb[0];
	this->pixels[(y * width + x) * this->numChannels + 1] = rgb[1];
	this->pixels[(y * width + x) * this->numChannels + 2] = rgb[2];
}


void Image::saveImage(std::string fileName)
{
	stbi_write_png(fileName.c_str(), this->width, this->height, this->numChannels, this->pixels.data(), this->width * this->numChannels * sizeof(char));
}


