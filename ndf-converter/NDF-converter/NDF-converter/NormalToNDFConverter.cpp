#include "NormalToNDFConverter.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>



NormalToNDFConverter::NormalToNDFConverter()
{

}

glm::vec2 NormalToNDFConverter::sampleRawNormal(Image& normalMap, int x, int y)
{
	int w = normalMap.width;
	int h = normalMap.height;

	x = glm::clamp(x, 0, w - 1);
	y = glm::clamp(y, 0, h - 1);

	char* color = normalMap.getPixel(x, y);
	return glm::vec2(color[0] / 255.0f, color[1] / 255.0f) * 2.0f - glm::vec2(1.0f);
}


glm::vec2 NormalToNDFConverter::sampleNormalMap(Image& normalMap, glm::vec2 uv)
{
	uv = glm::fract(uv);

	int w = normalMap.width;
	int h = normalMap.height;

	int x = uv.x * w;
	int y = uv.y * h;

	// Positioning of point relative to discrete pixels, needed for interpolation
	glm::vec2 st = glm::vec2((uv.x * w) - x, (uv.y * h) - y);

	glm::vec2 p1 = sampleRawNormal(normalMap, x, y);
	glm::vec2 p2 = sampleRawNormal(normalMap, x + 1, y);
	glm::vec2 l1 = glm::mix(p1, p2, st.x);

	glm::vec2 p3 = sampleRawNormal(normalMap, x, y + 1);
	glm::vec2 p4 = sampleRawNormal(normalMap, x + 1, y + 1);
	glm::vec2 l2 = glm::mix(p3, p4, st.x);

	return glm::mix(l1, l2, st.y);
}


glm::mat2 NormalToNDFConverter::sampleNormalMapJacobian(Image& normalMap, glm::vec2 uv)
{
	float hX = 1.f / normalMap.width;
	float hY = 1.f / normalMap.height;

	glm::vec2 xDiff = glm::vec2(0.f);

	// Sobel filter
	xDiff += sampleNormalMap(normalMap, uv + glm::vec2(hX, hY)) * 1.f;
	xDiff += sampleNormalMap(normalMap, uv + glm::vec2(hX, 0)) * 2.f;
	xDiff += sampleNormalMap(normalMap, uv + glm::vec2(hX, -hY)) * 1.f;

	xDiff += sampleNormalMap(normalMap, uv - glm::vec2(hX, hY)) * -1.f;
	xDiff += sampleNormalMap(normalMap, uv - glm::vec2(hX, 0)) * -2.f;
	xDiff += sampleNormalMap(normalMap, uv - glm::vec2(hX, -hY)) * -1.f;

	glm::vec2 yDiff;
	yDiff += sampleNormalMap(normalMap, uv + glm::vec2(hX, hY)) * 1.f;
	yDiff += sampleNormalMap(normalMap, uv + glm::vec2(0, hY)) * 2.f;
	yDiff += sampleNormalMap(normalMap, uv + glm::vec2(-hX, hY)) * 1.f;

	yDiff += sampleNormalMap(normalMap, uv - glm::vec2(hX, hY)) * -1.f;
	yDiff += sampleNormalMap(normalMap, uv - glm::vec2(0, hY)) * -2.f;
	yDiff += sampleNormalMap(normalMap, uv - glm::vec2(-hX, hY)) * -1.f;

	xDiff /= 8.f * hX;
	yDiff /= 8.f * hY;

	glm::mat2 J;
	J[0][0] = xDiff.x;
	J[0][1] = yDiff.x;

	J[1][0] = xDiff.y;
	J[1][1] = yDiff.y;
	return J;
}


void NormalToNDFConverter::generate4DNDF(int width, int height, float sigmaR)
{
	// Load normal map image from file
	Image normalMap = Image{ "../Data/normal3.png" };
	Image ndfImage = Image{ width, height };

	// Convert normal map to mixture of Gaussians landscape
	curvedElements4DNDF(ndfImage, normalMap, width, height, sigmaR);

	// Save P-NDF image
	ndfImage.saveImage("../Output/ndf.png");
}

void NormalToNDFConverter::curvedElements4DNDF(Image& ndfImage, Image& normalMap, int width, int height, float sigmaR)
{
	int mX = normalMap.width;
	int mY = normalMap.height;

	std::vector<Gaussian> gaussians;

	float h = 1.0f / mX;
	float sigmaH = h / glm::sqrt(8.0f * glm::log(2.0f));

	float sigmaH2 = sigmaH * sigmaH;
	float sigmaR2 = sigmaR * sigmaR;

	float invSigmaH2 = 1.f / sigmaH2;
	float invSigmaR2 = 1.f / sigmaR2;


	for (int i = 0; i < mX * mY; i++)
	{
		float x = (i % mX);
		float y = (i / mX);

		glm::vec2 u_i(x / (float)mX, y / (float)mY);

		Gaussian newGaussian;

		glm::vec2 normal = sampleNormalMap(normalMap, u_i);
		newGaussian.seedPoint = glm::vec4{ u_i.x, u_i.y, normal.x, normal.y };

		glm::mat2 jacobian = sampleNormalMapJacobian(normalMap, u_i);
		glm::mat2 transposedJacobian = glm::transpose(jacobian);

		// ===========================================================
		// Calculating inverse covariance matrix: Paper formula (14)
		// ===========================================================
		newGaussian.A = ((transposedJacobian * jacobian) * invSigmaR2) + glm::mat2(invSigmaH2);
		newGaussian.B = -transposedJacobian * invSigmaR2;
		newGaussian.C = glm::mat2(invSigmaR2);
		
		glm::mat4 invCov4D;

		// Upper left
		invCov4D[0][0] = newGaussian.A[0][0];
		invCov4D[0][1] = newGaussian.A[0][1];
		invCov4D[1][0] = newGaussian.A[1][0];
		invCov4D[1][1] = newGaussian.A[1][1];

		// Upper right
		invCov4D[2][0] = newGaussian.B[0][0];
		invCov4D[2][1] = newGaussian.B[0][1];
		invCov4D[3][0] = newGaussian.B[1][0];
		invCov4D[3][1] = newGaussian.B[1][1];

		// Lower left
		glm::mat2 trB = -jacobian * invSigmaR2;
		invCov4D[0][2] = trB[0][0];
		invCov4D[0][3] = trB[0][1];
		invCov4D[1][2] = trB[1][0];
		invCov4D[1][3] = trB[1][1];

		// Lower right
		invCov4D[2][2] = newGaussian.C[0][0];
		invCov4D[2][3] = newGaussian.C[0][1];
		invCov4D[3][2] = newGaussian.C[1][0];
		invCov4D[3][3] = newGaussian.C[1][1];

		newGaussian.invCov4d = invCov4D;

		float det = glm::determinant(glm::inverse(invCov4D) * 2.f * glm::pi<float>());

		if (det <= 0.0f)
		{
			newGaussian.coeff = 0.0f;
		}
		else {
			newGaussian.coeff = h * h / glm::sqrt(det);
		}

		gaussians.push_back(newGaussian);
	}

	// TODO: implement formula 12,13 to test instead of current evaluate
	Image testResult = { width, height };
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			glm::vec2 uv = { float(x) / float(width), float(y) / float(height) };
			char* normal = normalMap.getPixel(x, y);
			glm::vec3 normalVec3 = { int(normal[0]) / 255.0f, int(normal[1]) / 255.0f, int(normal[2]) / 255.0f };
			glm::vec2 st = { normalVec3.x, normalVec3.z };
			glm::vec4 pixelEvaluation = { uv.x, uv.y, st.x, st.y };

			float response = gaussians[32896].evaluate(pixelEvaluation);
			char responseColor[3] = { response * 255.0f, response * 255.0f, response * 255.0f };
			testResult.writePixel(x, y, responseColor);
		}
	}
	testResult.saveImage("../Data/testResult.png");

}
