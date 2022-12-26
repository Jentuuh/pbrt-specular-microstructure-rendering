#pragma once
#include "glm/glm.hpp"

class GaussianData
{
public:
	GaussianData();

	float evaluate(glm::vec4 x);
	float evaluateFormula12(glm::vec2 u, glm::vec2 s, float sigmaH2, float sigmaR2);

	glm::vec4 seedPoint; // position u, normal n(u)
	glm::mat2 A;
	glm::mat2 B;
	glm::mat2 C;
	glm::mat4 invCov4d;

	float coeff;
};

