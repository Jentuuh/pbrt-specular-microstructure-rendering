#pragma once
#include <glm/glm.hpp>

class Gaussian
{
public:
	Gaussian();

	float evaluate(glm::vec4 x);
	float evaluate(glm::vec2 x);

	glm::vec4 seedPoint; // position u, normal n(u)
	glm::mat2 A;
	glm::mat2 B;
	glm::mat2 C;
	glm::mat4 invCov4d;

	float coeff;
};

