#include <iostream>

#include "NormalToNDFConverter.h"
#include "Image.h"

int main()
{
    //Image test = Image{ "../Data/normal3.png" };
   /* Image test = Image{ 256, 256 };
    for (int i = 200; i < 250; i++)
    {
        for (int j = 200; j < 250; j++)
        {
            char pixel[3] = { 255, 255, 255 };
            test.writePixel(i, j, pixel);
        }
    }
    test.saveImage("../Data/test1.png");*/
    Image test = Image{"../Data/normal4.png", 3};
    char color[3] = { 255,255,255 };
    test.writePixel(0, 255, color);
    test.writePixel(1, 255, color);
    test.writePixel(2, 255, color);
    test.saveImage("../Data/testtt.png");

    int pixel[3];
    test.getPixel(192, 246, pixel);

    std::cout << "R: " << pixel[0] << std::endl;
    std::cout << "G: " << pixel[1] << std::endl;
    std::cout << "B: " << pixel[2] << std::endl;

    //std::cout << "R: " << int(pixel[0]) << ", G: " << int(pixel[1])
    //    << ", B: " << int(pixel[2]) << std::endl;

    //NormalToNDFConverter converter;
    //converter.generate4DNDF(522, 522, 0.005f);
}

