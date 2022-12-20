#include "NormalToNDFConverter.h"
#include <pcg32.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <thread>



NormalToNDFConverter::NormalToNDFConverter()
{

}

inline
float EvaluateGaussian(float c, const glm::vec2& x, const glm::vec2& u, const glm::mat2& InvCov)
{
	float inner = glm::dot(x - u, InvCov * (x - u));
	return c * glm::exp(-.5f * inner);
}

inline
float GetGaussianCoefficient(const glm::mat2& InvCov)
{
	float det = glm::determinant(2.f * glm::pi<float>() * glm::inverse(InvCov));

	if (det > 0.f)
		return 1.f / glm::sqrt(det);

	return 0.f;
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
	Image normalMap = Image{ "../Data/normal4.png", 3 };
	Image ndfImage = Image{ width, height };

	// Convert normal map to mixture of Gaussians landscape
	curvedElements4DNDF(ndfImage, normalMap, width, height, sigmaR);

	// Save P-NDF image
	//ndfImage.saveImage("../Output/ndf.png");
}

void integrateCurvedElements(int threadNumber, int chunkHeight, int width, int height, int mX, int mY, std::vector<Gaussian> gaussians, float* ndf)
{
	glm::vec2 regionCenter = glm::vec2{ 50, 50 };
	glm::vec2 regionSize = glm::vec2{ 16, 16 };
	glm::vec2 from = regionCenter - regionSize * 0.5f;

	float footprintRadius = regionSize.x * 0.5f / (float)mX;
	float sigmaP = footprintRadius * 0.5f;

	glm::mat2 footprintCovarianceInv = glm::inverse(glm::mat2(sigmaP * sigmaP));
	glm::vec2 footprintMean = (from + regionSize * .5f) * glm::vec2(1.f / mX, 1.f / mY);

	pcg32 generator;
	generator.seed(14041956 + threadNumber * 127361);

	int samplesPerPixel = 8;

	int fromY = chunkHeight * threadNumber;
	int toY = glm::min(chunkHeight * (threadNumber + 1), height);

	float invW = 1.f / (float)width;
	float invH = 1.f / (float)height;

	for (int y = fromY; y < toY; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float accum = 0.f;
			float s = x * invW;
			float t = y * invH;

			// Shift s,t between [-1;1]
			glm::vec2 imageS = glm::vec2((s * 2.f) - 1.f, (t * 2.f) - 1.f);

			// Projection outside the disk
			if (glm::length(imageS) > .975f)
			{
				ndf[y * width + x] = 0.f;
				continue;
			}

			for (int sample = 0; sample < samplesPerPixel; sample++)
			{
				s = (x + generator.nextFloat()) * invW;
				t = (y + generator.nextFloat()) * invH;


				// For each gaussian in the region...
				for (int gX = from.x; gX < regionSize.x + from.x; gX++)
				{
					for (int gY = from.y; gY < regionSize.y + from.y; gY++)
					{
						Gaussian data = gaussians[gY * mX + gX];
						glm::vec4 gaussianSeed = data.seedPoint;

						// Difference in direction between seedpoint and sample, S - N(u_i)
						glm::vec2 S((s * 2.f) - 1.f, (t * 2.f) - 1.f);
						S = S - glm::vec2(gaussianSeed.z, gaussianSeed.w);

						// We reduce the 4D gaussian into 2D by fixing S, see appendix formula 18
						glm::mat2 invCov = data.A;
						glm::vec2 u0 = -((glm::inverse(data.A)) * data.B) * S;
						float inner = glm::dot(S, data.C * S) - glm::dot(u0, data.A * u0);
						float c = data.coeff * glm::exp(-0.5f * inner);

						// Calculate the resulting gaussian by multiplying Gp * Gi
						glm::mat2 resultInvCovariance = invCov + footprintCovarianceInv;
						glm::mat2 resultCovariance = glm::inverse(resultInvCovariance);
						glm::vec2 resultMean = resultCovariance * (invCov * u0 + footprintCovarianceInv * (footprintMean - glm::vec2(gaussianSeed.x, gaussianSeed.y)));

						float resultC = EvaluateGaussian(c, resultMean, u0, invCov) *
							EvaluateGaussian(GetGaussianCoefficient(footprintCovarianceInv), resultMean, footprintMean - glm::vec2(gaussianSeed.x, gaussianSeed.y), footprintCovarianceInv);

						float det = (glm::determinant(resultCovariance * 2.f * glm::pi<float>()));

						if (det > 0.f)
							accum += resultC * glm::sqrt(det);
					}
				}
			}
			// Monte carlo integration
			accum /= (mX / (float)regionSize.x) * .8f;
			accum /= samplesPerPixel;

			ndf[y * width + x] = accum;
		}
		std::cout << "Processed row " << y << "..." << std::endl;
	}
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

	int threadCount = 16;
	std::vector<std::thread> threads;
	int chunkHeight = height / threadCount;

	// Shared output buffer
	float* ndf = new float[width * height];

	// Start threads
	for (int i = 0; i < threadCount; i++)
	{
		threads.push_back(std::thread{ integrateCurvedElements, i, chunkHeight, width, height, mX, mY, gaussians, ndf});
	}

	// Join threads
	for (int i = 0; i < threadCount; i++)
	{
		threads[i].join();
	}

	// Write output to image
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float normal = ndf[y * width + x];
			char color[3] = { glm::clamp(normal * 255.0f, 0.0f, 255.0f), glm::clamp(normal * 255.0f, 0.0f, 255.0f), glm::clamp(normal * 255.0f, 0.0f, 255.0f) };
			ndfImage.writePixel(x, y, color);
		}
	}

	ndfImage.saveImage("../Output/ndfImage.png");

	//Image testResult = { width, height };
	//for (int y = 0; y < width; y++)
	//{
	//	for (int x = 0; x < height; x++)
	//	{
	//		glm::vec2 uv = { float(x) / float(width), float(y) / float(height) };
	//		glm::vec2 st = sampleNormalMap(normalMap, uv);
	//		glm::vec4 pixelEvaluation = { uv.x, uv.y, st.x, st.y };

	//		float response = gaussians[131150].evaluateFormula12(uv, st, sigmaH2, sigmaR2);
	//		char responseColor[3] = { response * 255.0f, response * 255.0f, response * 255.0f };
	//		testResult.writePixel(x, y, responseColor);
	//	}
	//}
	//testResult.saveImage("../Data/testResult.png");
	delete[] ndf;
}
