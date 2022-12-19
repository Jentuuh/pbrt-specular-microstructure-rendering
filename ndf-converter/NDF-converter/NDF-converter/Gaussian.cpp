#include "Gaussian.h"

#include <iostream>
Gaussian::Gaussian(){}


float Gaussian::evaluate(glm::vec4 x)
{
	float inner = glm::dot(x - this->seedPoint, this->invCov4d * (x - this->seedPoint));
	return this->coeff * glm::exp(-.5f * inner);
}

float evaluate(glm::vec2 x)
{

}

