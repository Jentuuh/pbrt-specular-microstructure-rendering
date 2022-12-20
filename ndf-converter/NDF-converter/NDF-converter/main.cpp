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
    NormalToNDFConverter converter;
    converter.generate4DNDF(522, 522, 0.005f);
}

