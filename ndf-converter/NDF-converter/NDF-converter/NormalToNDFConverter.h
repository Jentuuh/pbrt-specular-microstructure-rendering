#pragma once
#include "Image.h"
#include "Gaussian.h"

class NormalToNDFConverter
{
public:
	NormalToNDFConverter();

	glm::vec2 sampleRawNormal(Image& normalMap, int x, int y);
	glm::vec2 sampleNormalMap(Image& normalMap, glm::vec2 uv);
	glm::mat2 sampleNormalMapJacobian(Image& normalMap, glm::vec2 uv);
	
	//void integrateCurvedElements(int threadNumber, int chunkHeight, int width, int height, int mX, int mY, std::vector<Gaussian> gaussians, float* ndf);
	void generate4DNDF(int width, int height, float sigmaR);
	void curvedElements4DNDF(Image& ndfImage, Image& normalMap, int width, int height, float sigmaR);
private:

};

