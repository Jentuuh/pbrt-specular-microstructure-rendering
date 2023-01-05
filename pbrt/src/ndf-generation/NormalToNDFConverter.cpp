#include "NormalToNDFConverter.h"
#include <pcg32.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <thread>
#include <random>


NormalToNDFConverter::NormalToNDFConverter(Image normalMap) : normalMap{normalMap} {
	// Initialize gaussian curved elements
    generateGaussianCurvedElements(0.0065f);
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

/**
* Helper function to transform a vector to another basis
*/
glm::vec3 NormalToNDFConverter::changeBasisTo(glm::vec3 in, glm::vec3 X,
                                            glm::vec3 Y, glm::vec3 Z) {
        return glm::vec3{glm::dot(in, X), glm::dot(in, Y), glm::dot(in, Z)};
}

glm::vec2 NormalToNDFConverter::sampleRawNormal(Image& normalMap, int x, int y)
{
	int w = normalMap.width;
	int h = normalMap.height;

	x = glm::clamp(x, 0, w - 1);
	y = glm::clamp(y, 0, h - 1);

    int color[3];
    normalMap.getPixel(x, y, color);
	
	// Normals are between [-1; 1] in both dimensions (projected onto unit disk)
	return glm::vec2(float(color[0]) / 255.0f, float(color[1]) / 255.0f) * 2.0f - glm::vec2(1.0f);
}


glm::vec2 NormalToNDFConverter::sampleNormalMap(Image& normalMap, glm::vec2 uv)
{
	uv = glm::fract(uv);

	int w = normalMap.width;
	int h = normalMap.height;

	int x = uv.x * w;
	int y = uv.y * h;

	// Fractional positioning of point relative to discrete pixels, needed for interpolation
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

	// 2D Sobel filter
	// https://en.wikipedia.org/wiki/Sobel_operator
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

/**
* Returns D(h) the response for position-normal distribution function. (The probability that a given half-vector corresponds 
* to a normal at that certain location in the normal map).
*/
float NormalToNDFConverter::evaluatePNDF(glm::vec2 U, glm::vec2 st,
                                         float sigmaR, int regionSize) {
			int mX = normalMap.width;
			int mY = normalMap.height;

			// Decide footprint boundary that defines the area of Gaussian elements that we should integrate over
			glm::vec2 regionCenter = glm::vec2{
                            U.x * float(mX), U.y * float(mY)};
			glm::vec2 regionSizeVec = glm::vec2(regionSize, regionSize);
            glm::vec2 from = glm::clamp(regionCenter - regionSizeVec * .5f, glm::vec2(), glm::vec2(mX - 1, mY - 1));
            glm::vec2 to = glm::clamp(regionCenter + regionSizeVec * .5f, glm::vec2(), glm::vec2(mX - 1, mY - 1));

			// Model pixel footprint area as a 2D-Gaussian instead of a square area
			float footprintRadius = regionSizeVec.x * .5f / (float)mX;
			float sigmaP = footprintRadius * 0.0625f;
			glm::mat2 footprintCovarianceInv =
				glm::inverse(glm::mat2(sigmaP * sigmaP));
			glm::vec2 footprintMean = (from + regionSizeVec * .5f) *
                                                  glm::vec2(1.f / mX, 1.f / mY);

			// Samples that fall outside of the P-NDF disk evaluate to 0
            if (glm::length(st) > .975f) {
                return 0.f;
            }

			// ====================================================================================
			// Calculate footprint normal distribution function (D_P(s)) response, paper formula 11
			// ====================================================================================
			float accum = 0.f;
			// Loop over the Gaussians in the footprint P
			for (int gY = from.y; gY < to.y; gY++) {
				for (int gX = from.x; gX < to.x; gX++) {
					GaussianData data = gaussians[gY * mX + gX];
					glm::vec4 gaussianSeed = data.seedPoint;

					// Direction, S - N(Xi)
					glm::vec2 S(st);
					S = S - glm::vec2(gaussianSeed.z, gaussianSeed.w);
					
					// ==========================================================
                    // Closed form solution of integral of 2D Gaussian (Appendix)
                    // ==========================================================
					// We reduce the 4D gaussian into 2D by fixing S, see appendix
					glm::mat2 invCov = data.A;
					glm::vec2 u0 = -((glm::inverse(data.A)) * data.B) * S;
					float inner =
						glm::dot(S, data.C * S) - glm::dot(u0, data.A * u0);
					float c = data.coeff * glm::exp(-0.5f * inner);
                   
					// Calculate the resulting gaussian by multiplying Gp * Gi
					glm::mat2 resultInvCovariance = invCov + footprintCovarianceInv;
					glm::mat2 resultCovariance = glm::inverse(resultInvCovariance);
					glm::vec2 resultMean =
						resultCovariance *
						(invCov * u0 +
						 footprintCovarianceInv *
							 (footprintMean -
							  glm::vec2(gaussianSeed.x, gaussianSeed.y)));
					
					// ====================================================
					// Calculating scaling coefficient c, Paper formula 20
					// ====================================================
					float resultC =
						EvaluateGaussian(c, resultMean, u0, invCov) *
						EvaluateGaussian(
							GetGaussianCoefficient(footprintCovarianceInv),
							resultMean,
							footprintMean -
								glm::vec2(gaussianSeed.x, gaussianSeed.y),
							footprintCovarianceInv);
					
					// ============================================================================
					// Closed form solution of integral of multivariate Gaussian (paper formula 21)
					// ============================================================================
					float det = (glm::determinant(resultCovariance * 2.f *
												  glm::pi<float>()));
					
					if (det > 0.f) {
                        accum += resultC * glm::sqrt(det);
                    }
				}
			}
        
			accum /= (mX / (float)regionSize);
               
			return accum;
}

/**
* Samples the direction that the ray should bounce to.
* Step 2 of the algorithm, as explained in the video (https://dl.acm.org/doi/10.1145/2897824.2925915)
*/
glm::vec3 NormalToNDFConverter::sampleWh(glm::vec2 U, int regionSize)
{
		// Describe 'pixel footprint'
		glm::vec2 regionCenter = glm::vec2{U.x * normalMap.width, U.y * normalMap.height};
		glm::vec2 regionSizeVec = glm::vec2{regionSize, regionSize};
		glm::vec2 regionOrigin = regionCenter - (0.5f * regionSizeVec);

		// Generate random point U in the footprint
		float randomXOffset = ((float)rand() / RAND_MAX);
		float randomYOffset = ((float)rand() / RAND_MAX);
		glm::ivec2 randomPointU = glm::ivec2 {glm::clamp(int(regionOrigin.x + regionSize * randomXOffset), 0, normalMap.width - 1), 
											glm::clamp(int(regionOrigin.y + regionSize * randomYOffset), 0, normalMap.width - 1)};
	
		// Find Gaussian covering random point U
		GaussianData sampledGaussian = gaussians[randomPointU.y * normalMap.width + randomPointU.x];

		// Seedpoint density + std. dev. of seedpoint gaussians
		float h = 1.0f / normalMap.width;
		float sigmaH = h / glm::sqrt(8.0f * glm::log(2.0f));
		std::random_device rd{};
		std::mt19937 gen{rd()};

		// Sample random normal from the Gaussian covering random point U
		// x^2 + y^2 + z^2 = 1 --> z = sqrt(1 - x^2 - y^2) (we changed 1.0f to 
		// 1.01f to prevent rounding errors resulting in negative zSquared values)
		float zSquared =
			1.01f - (sampledGaussian.seedPoint.z * sampledGaussian.seedPoint.z) -
			(sampledGaussian.seedPoint.w * sampledGaussian.seedPoint.w);
        float z = sqrtf(zSquared);

		std::normal_distribution<float> samplenormalX{sampledGaussian.seedPoint.z, sigmaH};
		std::normal_distribution<float> samplenormalY{sampledGaussian.seedPoint.w, sigmaH};
		std::normal_distribution<float> samplenormalZ{z, sigmaH};

		float sampledX = samplenormalX(gen);
		float sampledY = samplenormalY(gen);
		float sampledZ = samplenormalZ(gen);

		glm::vec3 new_Wh = glm::vec3{sampledX, sampledY, sampledZ};
		return glm::normalize(new_Wh);
}

/**
* Generates Gaussian curved elements for the normal map that this converter was given. Gaussian elements provide a notion of the local position-normal 
* distribution at a certain coordinate in the normal map.
*/
void NormalToNDFConverter::generateGaussianCurvedElements(float sigmaR) {
    int mX = normalMap.width;
    int mY = normalMap.height;

    // ==========
	// Constants
    // ==========
    const float h = 1.0f / mX;
    const float sigmaH = h / glm::sqrt(8.0f * glm::log(2.0f));	// std. dev. of seed gaussians

    const float sigmaH2 = sigmaH * sigmaH;
    const float sigmaR2 = sigmaR * sigmaR;

    const float invSigmaH2 = 1.f / sigmaH2;
    const float invSigmaR2 = 1.f / sigmaR2;

    for (int i = 0; i < mX * mY; i++) {
        float x = (i % mX);
        float y = (i / mX);

        glm::vec2 u_i(x / (float)mX, y / (float)mY);

        GaussianData newGaussian;

        glm::vec2 normal = sampleNormalMap(normalMap, u_i);

        newGaussian.seedPoint = glm::vec4{u_i.x, u_i.y, normal.x, normal.y};

        glm::mat2 jacobian = sampleNormalMapJacobian(normalMap, u_i);
        glm::mat2 transposedJacobian = glm::transpose(jacobian);

        // ===========================================================
        // Calculating inverse covariance matrix: Paper formula (14)
        // ===========================================================
        newGaussian.A = ((transposedJacobian * jacobian) * invSigmaR2) +
                        glm::mat2(invSigmaH2);
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

        float det = glm::determinant(glm::inverse(invCov4D) * 2.f *
                                        glm::pi<float>());

		// Covariance matrix needs to be invertible and cannot change orientation (formula 10 paper)
        if (det <= 0.0f || isnan(det)) {
            newGaussian.coeff = 0.0f;
        } else {
            newGaussian.coeff = h * h / glm::sqrt(det);
        }

        gaussians.push_back(newGaussian);
    }
}