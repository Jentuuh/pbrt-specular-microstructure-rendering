#pragma once
#include "Image.h"
#include "GaussianData.h"


class NormalToNDFConverter
{
public:
	NormalToNDFConverter(Image normalMap);

	glm::vec2 sampleRawNormal(Image& normalMap, int x, int y);
	glm::vec2 sampleNormalMap(Image& normalMap, glm::vec2 uv);
	glm::mat2 sampleNormalMapJacobian(Image& normalMap, glm::vec2 uv);
	
	float evaluatePNDF(glm::vec2 U, glm::vec2 st, float sigmaR);
	void generateGaussianCurvedElements(float sigmaR);
	void generate4DNDF(int width, int height, float sigmaR);
	void curvedElements4DNDF(Image& ndfImage, Image& normalMap, int width, int height, float sigmaR);
private:
	Image normalMap;
	std::vector<GaussianData> gaussians;
};

