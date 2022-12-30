#pragma once
#include "Image.h"
#include "GaussianData.h"


class NormalToNDFConverter
{
public:
	NormalToNDFConverter(Image normalMap);

	glm::vec3 changeBasisTo(glm::vec3 in, glm::vec3 X, glm::vec3 Y,
                                glm::vec3 Z);
	glm::vec2 sampleRawNormal(Image& normalMap, int x, int y);
	glm::vec2 sampleNormalMap(Image& normalMap, glm::vec2 uv);
	glm::mat2 sampleNormalMapJacobian(Image& normalMap, glm::vec2 uv);
	
	float evaluatePNDF(glm::vec2 U, glm::vec2 st, float sigmaR, int regionSize);
    glm::vec3 sampleWh(glm::vec2 U, int regionSize);
	void generateGaussianCurvedElements(float sigmaR);
	void generate4DNDF(int width, int height, float sigmaR);
	void curvedElements4DNDF(Image& ndfImage, Image& normalMap, int width, int height, float sigmaR);
private:
	Image normalMap;
	std::vector<GaussianData> gaussians;
};

