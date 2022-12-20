#include "Gaussian.h"

#include <iostream>
Gaussian::Gaussian(){}


float Gaussian::evaluate(glm::vec4 x)
{
	float inner = glm::dot(x - this->seedPoint, this->invCov4d * (x - this->seedPoint));
	return this->coeff * glm::exp(-.5f * inner);
}

float Gaussian::evaluate(glm::vec2 x)
{

}

float Gaussian::evaluateFormula12(glm::vec2 u, glm::vec2 s, float sigmaH2, float sigmaR2)
{
	float inner1 = -powf(glm::length((u - glm::vec2{ this->seedPoint.x, this->seedPoint.y })), 2) / (2.0f * sigmaH2);
	float inner2 = -powf(glm::length((s - glm::vec2{ this->seedPoint.z, this->seedPoint.w })), 2) / (2.0f * sigmaR2);
	return this->coeff * glm::exp(inner1) * glm::exp(inner2);
}
