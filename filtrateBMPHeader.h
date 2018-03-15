#pragma once
namespace filters {
	__declspec(dllexport) 	unsigned char* filtrateRGBMatrix(unsigned char* tabRGB, int imgWidth, int imgHeight, double *filter, int filterWidth, int filterHeight);
}