#include "Image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <iostream>

#include "stb_image.h"
// test
Image::Image(std::string filename, int imageChannels) {
    int normalWidth;
    int normalHeight;
    int numChannels;

    unsigned char* normalMap =
        stbi_load(filename.c_str(), &normalWidth, &normalHeight, &numChannels,
                  imageChannels);
    this->pixels = std::vector<char>(
        normalMap, normalMap + (normalWidth * normalHeight * numChannels));
    this->width = normalWidth;
    this->height = normalHeight;
    this->numChannels = numChannels;

    std::cout << "Loaded image of dimensions: (" << normalWidth << ","
              << normalHeight << "," << numChannels << ")" << std::endl;
}

Image::Image(int width, int height) {
    this->pixels = std::vector<char>(width * height * 3, 0.0f);
    this->width = width;
    this->height = height;
    this->numChannels = 3;
}

int* Image::getPixel(int x, int y, int* pixel) {
    unsigned bytePerPixel = numChannels;
    unsigned char r = pixels[(y * width + x) * bytePerPixel];
    unsigned char g = pixels[(y * width + x) * bytePerPixel + 1];
    unsigned char b = pixels[(y * width + x) * bytePerPixel + 2];
    unsigned char a =
        numChannels >= 4 ? pixels[(y * width + x) * bytePerPixel + 3] : 0xff;

    pixel[0] = int(r);
    pixel[1] = int(g);
    pixel[2] = int(b);
    pixel[3] = int(a);

    return pixel;
}

// `rgb_a` should be char[3] or char[4], depending on the amount of channels in
// the image
void Image::writePixel(int x, int y, char* rgb_a) {
    this->pixels[(y * width + x) * this->numChannels] = rgb_a[0];
    this->pixels[(y * width + x) * this->numChannels + 1] = rgb_a[1];
    this->pixels[(y * width + x) * this->numChannels + 2] = rgb_a[2];
    if (numChannels >= 4) {
        this->pixels[(y * width + x) * this->numChannels + 3] = rgb_a[3];
    }
}

void Image::saveImage(std::string fileName) {
    stbi_write_png(fileName.c_str(), this->width, this->height,
                   this->numChannels, this->pixels.data(),
                   this->width * this->numChannels * sizeof(char));
}
