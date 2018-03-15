#include <iostream>
#include <cstdlib>
#include <fstream>
#include <stddef.h>
#include "EasyBMP.cpp"
#include "EasyBMP_BMP.h"
#include "EasyBMP_DataStructures.h"
#include "filtrateBMPHeader.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <allegro5\allegro.h>
#include <allegro5\allegro_color.h>
#include <allegro5\allegro_primitives.h> 
#include <allegro5\allegro_windows.h>
#include <allegro5/allegro_image.h>

#define GREEN al_map_rgb(0, 255, 0)
#define RED ALLEGRO_COLOR(255, 0, 0)
#define BLACK al_map_rgb(0, 0, 0)
int ray = 20;
int x1 = 100;
//int y1 = 100;
int x2 = 300;
int y2 = 300;
int circle_x = x1;
int circle_y = 100;

using namespace std;

const unsigned int NUMBER_OF_THREADS = 397;

const int USE_CPU = 0;
const int USE_GPU = 1;

/*
Copyright (c) 2005, The EasyBMP Project (http://easybmp.sourceforge.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
   3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


__global__ void filtrateGPU(unsigned char *inputTab, unsigned char *outputTab, double* d_Filter,
	int *d_imgWidth, int *d_imgHeight, int *d_filterWidth, int *d_filterHeight,
	double *d_filterSum, int *d_extPrt) {
	
		int id = blockIdx.x *blockDim.x + threadIdx.x;
		if (id<=4**d_imgHeight**d_imgWidth) {
		int idf = 0;
		int f = 0;
		double filteredPixel = 0;
		for (int y = -*d_extPrt; y <= *d_extPrt; y++) {
			for (int x = -*d_extPrt; x <= *d_extPrt; x++) {
				idf = id + y**d_imgWidth + x;
				filteredPixel += (double)inputTab[idf] * d_Filter[f];
				f++;
			}
		}

		filteredPixel /= *d_filterSum;

		outputTab[id] = (unsigned char)filteredPixel;
	}
}


// ************************* filtry do przetwarzania obrazow ***************************


const int GAUSS_FILTER = 100;
const int MASKAFG_FILTER = 101;
const int USREDNIAJACY_FILTER = 102;
const int USREDNIAJACYDUZY_FILTER = 103;
const int WYOSTRZAJACY_FILTER = 104;
const int SOBELY_FILTER = 105;
const int SOBELX_FILTER = 106;
const int LAPL_FILTER = 107;
const int LAPLDUZY_FILTER = 108;
const int USUNSREDNIA_FILTER = 109;
const int HP2_FILTER = 110;
const int HP3_FILTER = 111;
const int UWYPUKLAJACY_FILTER = 112;
const int WSCHOD_FILTER = 113;
const int PIRAMIDALNYDUZY_FILTER = 114;

const int SKALASZAROSCI_CHOICE = 115;

double gauss[49] = { 1,	1,	2,	2,	2,	1,	1,
1,	2,	2,	4,	2,	2,	1,
2,	2,	4,	8,	4,	2,	2,
2,	4,	8, 16,	8,	4,	2,
2,	2,	4,	8,	4,	2,	2,
1,	2,	2,	4,	2,	2,	1,
1,	1,	2,	2,	2,	1,	1 };
int gauss_width = 7;
int gauss_height = 7;

double maskaFG[9] = { 1, -2,  1,
-2,  5, -2,
1, -2,  1 };
int maskFG_width = 3;
int maskaFG_height = 3;

double usredniajacy[9] = { 1, 1, 1,
1, 1, 1,
1, 1, 1 };
int usredniajacy_width = 3;
int usredniajacy_height = 3;

double usredniajacyDuzy[25] = { 1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1 };
int usredniajacyDuzy_width = 5;
int usredniajacyDuzy_height = 5;

double wyostrzajacy[9] = { 0, -2,  0,
-2, 17, -2,
0, -2,  0 };
int wyostrzajacy_width = 3;
int wyostrzajacy_height = 3;

double sobelY[9] = { 1,  2,  1,
0,  0,  0,
-1,  2, -1 };
int sobelY_width = 3;
int sobelY_height = 3;


double sobelX[9] = { -1,  0,  1,
2,  0,  2,
-1,  0,  1 };
int sobelX_width = 3;
int sobelX_height = 3;

double LAPL[9] = { 0, -1,  0,
-1 , 4, -1,
0, -1,  0 };
int LAPL_width = 3;
int LAPL_height = 3;

double LAPL_DUZY[25] = { -1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 24, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1 };
int LAPL_DUZY_width = 5;
int LAPL_DUZY_height = 5;

double usunSrednia[9] = { -1, -1, -1,
-1,  8, -1,
-1, -1, -1 };
int usunSrednia_width = 3;
int usunSrednia_height = 3;

double HP2[9] = { 1,  -2,   1,
-2,   5,  -2,
1,  -2,   1 };
int HP2_width = 3;
int HP2_height = 3;

double HP3[9] = { 0,  -1,  0,
-1,  20, -1,
0,  -1,  0 };
int HP3_width = 3;
int HP3_height = 3;

double uwypuklajacy[9] = { -1, 0, 1,
-1, 1, 1,
-1, 0, 1 };
int uwypuklajacy_width = 3;
int uwypuklajacy_height = 3;

double wschod[9] = { -1,  1,  1,
-1, -2,  1,
-1,  1,  1 };
int wschod_width = 3;
int wschod_height = 3;

double piramidalnyDuzy[25] = { 1, 2, 3, 2, 1,
2, 4, 6, 4, 2,
3, 6, 9, 6, 3,
2, 4, 6, 4, 2,
1, 2, 3, 2, 1 };
int piramidalnyDuzy_width = 5;
int piramidalnyDuzy_height = 5;

// *********************** Zwraca pixel z podanymi wartościami (RGBA) ***********************
RGBApixel assignPixel(double rVal, double gVal, double bVal, double aVal) {

	RGBApixel pixel;
	pixel.Red = rVal;
	pixel.Green = gVal;
	pixel.Blue = bVal;
	pixel.Alpha = aVal;

	return pixel;
}

// *********************** Zwraca pixel z podanymi wartościami (RGB) ***********************
RGBApixel assignPixel(double rVal, double gVal, double bVal) {

	RGBApixel pixel;
	pixel.Red = rVal;
	pixel.Green = gVal;
	pixel.Blue = bVal;

	return pixel;
}

// pixel w odcieniu szarości
RGBApixel assignPixel(double val) {

	return assignPixel(val, val, val);

}

// *********************** Zmienia głębokość bitmapy i tworzy nową,
//                         nie naruszając starej ***********************
void changeBMPbitDepth(char* imgName, char* newImageName, int bitDepth) {

	BMP bitMap;
	if (bitDepth == 1 || bitDepth == 4 || bitDepth == 8 ||
		bitDepth == 24 || bitDepth == 32) {
		bitMap.ReadFromFile(imgName);
		bitMap.SetBitDepth(bitDepth);
		bitMap.WriteToFile(newImageName);
	}
	else {
		cout << "Blad, nie udalo sie przekonwertowac! chngBMPbitDepthERR" << endl;
		system("Pause");
	}
	return;
}

// zmienia bitmape kolorową na czarnobiałą, nie naruszając starej (tworzy nową)
void changeToGrayScale(char* imgName, char* newImageName) {

	BMP bitmap;
	BMP filteredBitMap;

	bitmap.ReadFromFile(imgName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();

	filteredBitMap.SetSize(width, height);
	filteredBitMap.SetBitDepth(bitmap.TellBitDepth());

	RGBApixel pixelTMP;

	double red = 0;
	double blue = 0;
	double green = 0;
	double newColVal = 0;

	for (int y = 0; y < height; y++) {			// y
		for (int x = 0; x < width; x++) {		// x

			pixelTMP = bitmap.GetPixel(x, y);
			newColVal = (pixelTMP.Red + pixelTMP.Green + pixelTMP.Blue) / 3;
			pixelTMP = assignPixel(newColVal);

			filteredBitMap.SetPixel(x, y, pixelTMP);

		}
	}

	filteredBitMap.WriteToFile(newImageName);


	return;
}

// zwieksza roznice pomiedzy krawedziami, a reszta obrazu
void showGrayScaleEdges(char* imgName, char* newImageName) {

	BMP bitmap;
	BMP filteredBitMap;

	bitmap.ReadFromFile(imgName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();

	filteredBitMap.SetSize(width, height);
	filteredBitMap.SetBitDepth(bitmap.TellBitDepth());

	RGBApixel pixelTMP;

	double red = 0;
	double blue = 0;
	double green = 0;
	double newColVal = 0;

	for (int y = 0; y < height; y++) {			// y
		for (int x = 0; x < width; x++) {		// x

			pixelTMP = bitmap.GetPixel(x, y);
			if (pixelTMP.Red < 128)
				newColVal = pixelTMP.Red / 2;
			else {
				newColVal = pixelTMP.Red + 32;
			}

			pixelTMP = assignPixel(newColVal);

			filteredBitMap.SetPixel(x, y, pixelTMP);

		}
	}

	filteredBitMap.WriteToFile(newImageName);

	return;
}

// Uwidacznia krawedzie - eksperymentalne
void showBMPEdges(char* imgName, char* filteredName, char* edgedName) {

	BMP bitmap;
	BMP filteredbBitmap;
	BMP edgedBitmap;

	bitmap.ReadFromFile(imgName);
	filteredbBitmap.ReadFromFile(filteredName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();

	edgedBitmap.SetSize(width, height);
	edgedBitmap.SetBitDepth(bitmap.TellBitDepth());

	double red = 0;
	double blue = 0;
	double green = 0;
	double alpha = 0;

	for (int y = 0; y < height; y++) {				// y
		for (int x = 0; x < width; x++) {			// x

			red = 255;
			blue = 255;
			green = 255;
			alpha = 255;

			if (bitmap.GetPixel(x, y).Red != filteredbBitmap.GetPixel(x, y).Red) {
				red = 0;
			}
			if (bitmap.GetPixel(x, y).Green != filteredbBitmap.GetPixel(x, y).Green) {
				blue = 0;
			}
			if (bitmap.GetPixel(x, y).Blue != filteredbBitmap.GetPixel(x, y).Blue) {
				green = 0;
			}
			if (bitmap.GetPixel(x, y).Alpha != filteredbBitmap.GetPixel(x, y).Alpha) {
				alpha = 0;
			}
			edgedBitmap.SetPixel(x, y, assignPixel(red, green, blue, alpha));
		}
	}

	edgedBitmap.WriteToFile(edgedName);

	return;
}

// Odwraca kolory monochromatycznej bitmapy (jednobitowej)
void blackToWhite(char* imgName) {

	BMP bitmap;
	BMP wBitmap;

	bitmap.ReadFromFile(imgName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();

	wBitmap.SetSize(width, height);
	wBitmap.SetBitDepth(bitmap.TellBitDepth());

	double red = 0;
	double blue = 0;
	double green = 0;
	double alpha = 0;
	//************************* 0 - Black, 255 - White *************************
	for (int y = 1; y < height - 1; y++) {				// y
		for (int x = 1; x < width - 1; x++) {				// x

			red = 0;
			blue = 0;
			green = 0;
			alpha = 0;

			RGBApixel pixel;
			pixel = bitmap.GetPixel(x, y);

			if (pixel.Red == 0) {
				red = 255;
			}
			if (pixel.Green == 0) {
				green = 255;
			}
			if (pixel.Blue == 0) {
				blue = 255;
			}
			if (pixel.Alpha == 0) {
				alpha = 255;
			}

			wBitmap.SetPixel(x, y, assignPixel(red, green, blue, alpha));
		}
	}

	wBitmap.WriteToFile(imgName);

	return;
}

// *************************Właściwa filtracja (CPU)*************************
unsigned char* filtrateRGBMatrixCPU(unsigned char* tabRGB, int imgWidth, int imgHeight, double *filter, int filterWidth, int filterHeight) {

	// Tworzy dwie tablice
	unsigned char *tabFil = new unsigned char[4 * imgHeight * imgWidth];	//Output TAB
																			/*for (int y = 0; y < 4 * imgHeight; y++) {
																			tabFil[y] = new unsigned char[imgWidth];
																			}*/
	double filteredPixelR;
	double filteredPixelG;
	double filteredPixelB;
	double filteredPixelA;

	double filterSum = 0;
	for (int i = 0; i < filterWidth*filterHeight; i++) {
		filterSum += filter[i];
	}

	if (filterSum == 0) {
		filterSum = 1;
	}


	int extPrt;			// ile pixeli na prawo (lub lewo) od środkowego, externalPart
	extPrt = filterWidth / 2;

	int f = 0;			// indeks z tablicy filtru (filter[9])
	double newValR = 0;	// nowa wart pixela
	double newValG = 0;
	double newValB = 0;
	double newValA = 0;



	//petla FILTRUJE wszystkie pixele za wyjatkiem obwódki o szer 1
	// naszym elementem bedzie i*imgWidth + j
	for (int i = extPrt; i < imgHeight - extPrt; i++) {			// dla kazdego wiersza
		for (int j = extPrt; j < imgWidth - extPrt; j++) {		// dla kazdego elementu wiersza

			f = 0;
			newValR = 0;
			newValG = 0;
			newValB = 0;
			newValA = 0;

			for (int y = i - extPrt; y <= i + extPrt; y++) {
				for (int x = j - extPrt; x <= j + extPrt; x++) {
					newValR += ((double)tabRGB[imgWidth * y + x]) * filter[f];
					newValG += ((double)tabRGB[(imgHeight * imgWidth) + imgWidth * y + x]) * filter[f];
					newValB += ((double)tabRGB[2 * (imgHeight * imgWidth) + imgWidth * y + x]) * filter[f];
					newValA += ((double)tabRGB[3 * (imgHeight * imgWidth) + imgWidth * y + x]) * filter[f];
					f++;
				}
			}

			filteredPixelR = newValR / filterSum;
			filteredPixelG = newValG / filterSum;
			filteredPixelB = newValB / filterSum;
			filteredPixelA = newValA / filterSum;

			tabFil[imgWidth*i + j] = (unsigned char)filteredPixelR;
			tabFil[(imgHeight*imgWidth) + imgWidth*i + j] = (unsigned char)filteredPixelG;
			tabFil[2 * (imgHeight*imgWidth) + imgWidth*i + j] = (unsigned char)filteredPixelB;
			tabFil[3 * (imgHeight*imgWidth) + imgWidth*i + j] = (unsigned char)filteredPixelA;

		}


	}



	return tabFil;
}

// *************************Właściwa filtracja (GPU)*************************
unsigned char* filtrateRGBMatrixGPU(unsigned char* tabRGB, int imgWidth, int imgHeight, double *filter, int filterWidth, int filterHeight) {

	// Tworzy tablice
	unsigned char *tabFil = new unsigned char[4 * imgHeight * imgWidth];

	double filteredPixelR;
	double filteredPixelG;
	double filteredPixelB;
	double filteredPixelA;

	double filterSum = 0;
	for (int i = 0; i < filterWidth*filterHeight; i++) {
		filterSum += filter[i];
	}

	if (filterSum == 0) {
		filterSum = 1;
	}


	int extPrt;			// ile pixeli na prawo (lub lewo) od środkowego, externalPart
	extPrt = filterWidth / 2;

	int f = 0;			// indeks z tablicy filtru (filter[9])
	double newValR = 0;	// nowa wart pixela
	double newValG = 0;
	double newValB = 0;
	double newValA = 0;

	//GPU stuff ------- DEVICE'S POINTERS
	unsigned char *d_tabRGB;
	unsigned char *d_tabFil;
	double *d_Filter;

	int *d_imgWidth, *d_imgHeight,
		*d_filterWidth, *d_filterHeight;
	
	double *d_filterSum;
	int *d_extPrt;

	// ********************************************************Allocate********************************************************
	// **************Allocate InputTab, OutputTab, FilterTab, imgWidth, imgHeight, filterWidth, filterHeight******************
	// **************filterSum******************
	if (cudaMalloc(&d_tabRGB, sizeof(unsigned char)* 4 * imgHeight * imgWidth) != cudaSuccess) {
		cout << "Nope! a";
		return 0;
	}

	if (cudaMalloc(&d_tabFil, sizeof(unsigned char)* 4 * imgHeight * imgWidth) != cudaSuccess) {
		cout << "Nope! b";
		cudaFree(d_tabRGB);
		return 0;
	}

	if (cudaMalloc(&d_Filter, sizeof(double) * filterWidth * filterHeight) != cudaSuccess) {
		cout << "Nope! c";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);

		return 0;
	}

	if (cudaMalloc(&d_imgWidth, sizeof(int)) != cudaSuccess) {
		cout << "Nope! d";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);

		return 0;
	}

	if (cudaMalloc(&d_imgHeight, sizeof(int)) != cudaSuccess) {
		cout << "Nope! e";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);

		return 0;
	}

	if (cudaMalloc(&d_filterWidth, sizeof(int)) != cudaSuccess) {
		cout << "Nope! f";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);

		return 0;
	}

	if (cudaMalloc(&d_filterHeight, sizeof(int)) != cudaSuccess) {
		cout << "Nope! g";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		
		return 0;
	}

	if (cudaMalloc(&d_filterSum, sizeof(double)) != cudaSuccess) {
		cout << "Nope! h";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);

		return 0;
	}

	if (cudaMalloc(&d_extPrt, sizeof(int)) != cudaSuccess) {
		cout << "Nope! i";
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);

		return 0;
	}


	// ******************************************************** COPY ********************************************************
	// ***************** Copy From Host to Device  (Except filtrated Tab)******************************
	if (cudaMemcpy(d_tabRGB, tabRGB, sizeof(unsigned char)* 4 * imgHeight * imgWidth, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_Filter, filter, sizeof(double) * filterHeight * filterWidth, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_imgWidth, &imgWidth, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_imgHeight, &imgHeight, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_filterWidth, &filterWidth, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_filterHeight, &filterHeight, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_filterSum, &filterSum, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	if (cudaMemcpy(d_extPrt, &extPrt, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	// main part, Filtration TO DO FROM HERE <------------------------------------------------------------------------->
	int blocksOfThreads = 4 * imgHeight*imgWidth/ NUMBER_OF_THREADS;
	filtrateGPU << <blocksOfThreads, NUMBER_OF_THREADS >> > (d_tabRGB, d_tabFil, d_Filter, d_imgWidth, d_imgHeight,
		d_filterWidth, d_filterHeight, d_filterSum, d_extPrt);
	

	// ************************************************* COPY BACK*************************************************
	if (cudaMemcpy(tabFil, d_tabFil, sizeof(unsigned char) * 4 * imgHeight * imgWidth, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Could not copy! " << endl;
		cudaFree(d_tabRGB);
		cudaFree(d_tabFil);
		cudaFree(d_Filter);
		cudaFree(d_imgWidth);
		cudaFree(d_imgHeight);
		cudaFree(d_filterWidth);
		cudaFree(d_filterHeight);
		cudaFree(d_filterSum);
		cudaFree(d_extPrt);
		return 0;
	}

	// ******************************************************** FREE ********************************************************
	
	cudaFree(d_tabRGB);
	cudaFree(d_tabFil);
	cudaFree(d_Filter);
	cudaFree(d_imgWidth);
	cudaFree(d_imgHeight);
	cudaFree(d_filterWidth);
	cudaFree(d_filterHeight);
	cudaFree(d_filterSum);
	cudaFree(d_extPrt);

	return tabFil;
}

// Funkca "Opakowujaca filtracje"
void filtrateBMP(char* imgName, char* filteredImageName,
	double *filter, int filterWidth, int filterHeight) {

	BMP bitmap;
	BMP filteredBitMap;

	bitmap.ReadFromFile(imgName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();

	filteredBitMap.SetSize(width, height);
	filteredBitMap.SetBitDepth(bitmap.TellBitDepth());

	RGBApixel pixelTMP;

	int extPrt;			// ile pixeli na prawo (lub lewo) od środkowego, externalPart
	extPrt = filterWidth / 2;

	//******************** BMP -> Unsigned Char[][] ********************

	// Tworzy dwie tablice
	unsigned char *tabRGB = new unsigned char[4 * height*width];	//Input  TAB
	unsigned char *tabFil;											//Output TAB
																/*for (int y = 0; y < 4 * height; y++) {
																tabRGB[y] = new unsigned char[width];
																}*/

																// Wypełnia We tablice
	for (int y = 0; y < height; y++) {				// y
		for (int x = 0; x < width; x++) {			// x
			pixelTMP = bitmap.GetPixel(x, y);
			tabRGB[width*y + x] = (unsigned char)pixelTMP.Red;
			tabRGB[(height*width) + width*y + x] = (unsigned char)pixelTMP.Green;
			tabRGB[2 * (height*width) + width*y + x] = (unsigned char)pixelTMP.Blue;
			tabRGB[3 * (height*width) + width*y + x] = (unsigned char)pixelTMP.Alpha;

		}
	}

	//*************************** MAIN PART HERE **********************************
	tabFil = filtrateRGBMatrixGPU(tabRGB, width, height, filter, filterWidth, filterHeight);		// Filtracja tablicy RGB
	//tabFil = filtrateRGBMatrixCPU(tabRGB, width, height, filter, filterWidth, filterHeight);		// Filtracja tablicy RGB
	
	//  ************************ unsigned char[][] - > BMP ************************
	for (int y = extPrt; y < height - extPrt; y++) {
		for (int x = extPrt; x < width - extPrt; x++) {
			pixelTMP = assignPixel(tabFil[width*y + x], tabFil[(height*width) + width*y + x], tabFil[2 * (height*width) + width*y + x], tabFil[3 * (height*width) + width*y + x]);
			filteredBitMap.SetPixel(x, y, pixelTMP);
		}
	}

	filteredBitMap.WriteToFile(filteredImageName);

	//***************************** Delete TABs ********************************

	delete[] tabRGB;
	delete[] tabFil;

	tabRGB = NULL;
	tabFil = NULL;

	return;
}


// Usuwanie Szumów
// - usuwua tzw. pojedyncze czarne pixele,
// - po filtracji z wykrywaniem krawędzi
void removeNoise(char* imgName, char* filteredImageName, unsigned int redox) {

	BMP bitmap;
	BMP filteredBitMap;

	bitmap.ReadFromFile(imgName);

	const int width = bitmap.TellWidth();
	const int height = bitmap.TellHeight();


	RGBApixel pixelTMP;

	int extPrt = 1;
	int f = 0;

	RGBApixel blackPix;
	blackPix.Red = 0;
	blackPix.Green = 0;
	blackPix.Blue = 0;
	blackPix.Alpha = 0;

	RGBApixel whitePix;
	whitePix.Red = 255;
	whitePix.Green = 255;
	whitePix.Blue = 255;
	whitePix.Alpha = 0;


	for (int j = 1; j < height - 1; j++) {			// y
		for (int i = 1; i < width - 1; i++) {			// x

			f = 0;

			if (bitmap.GetPixel(i, j).Red == blackPix.Red) {

				bitmap.SetPixel(i, j, whitePix);

				for (int y = j - 1; y <= j + 1; y++) {
					for (int x = i - 1; x <= i + 1; x++) {

						if (bitmap.GetPixel(x, y).Red == 0) {
							f++;
							if (f >= redox) {
								bitmap.SetPixel(i, j, blackPix);
							}
						}


					}
				}
			}



		}


	}

	bitmap.WriteToFile(filteredImageName);

}

void action() {
	int choice = UWYPUKLAJACY_FILTER;

	switch (choice) {
	case GAUSS_FILTER:
		filtrateBMP("image.bmp", "image_filtered.bmp", gauss, gauss_width, gauss_height);
		break;
	case MASKAFG_FILTER:
		break;
	case USREDNIAJACY_FILTER:
		break;
	case USREDNIAJACYDUZY_FILTER:
		break;
	case WYOSTRZAJACY_FILTER:
		break;
	case SOBELY_FILTER:
		break;
	case SOBELX_FILTER:
		break;
	case LAPL_FILTER:
		break;
	case LAPLDUZY_FILTER:
		break;
	case USUNSREDNIA_FILTER:
		break;
	case HP2_FILTER:
		break;
	case HP3_FILTER:
		break;
	case UWYPUKLAJACY_FILTER:
		filtrateBMP("samuraj.bmp", "filteredSamuraj.bmp", uwypuklajacy, uwypuklajacy_width, uwypuklajacy_height);
		break;
	case WSCHOD_FILTER:
		break;
	case PIRAMIDALNYDUZY_FILTER:
		break;
	case SKALASZAROSCI_CHOICE:
		break;
	default:
		break;
	}
}

int main() {

	//al_init(); // inicjowanie biblioteki allegro
	//al_install_keyboard(); // instalowanie sterownika klawiatury
	//al_init_image_addon();// inicjowanie dodatku umożliwiającego odczyt jak i zapis obrazów, w formatach BMP,  PNG, JPG, PCX, TGA.
	//ALLEGRO_KEYBOARD_STATE klawiatura; // utworzenie struktury do odczytu stanu klawiatury
	//ALLEGRO_DISPLAY *okno = al_create_display(320, 240);// tworzymy wskaźnik okna, i podajemy jego szer. i wys
	//al_set_window_title(okno, "Allegro5 kurs pierwsze okno");// podajemy tytuł okna
	//ALLEGRO_BITMAP *obrazek = al_load_bitmap("samuraj.bmp");// wczytujemy bitmapę do pamięci
	//while (!al_key_down(&klawiatura, ALLEGRO_KEY_ESCAPE)) //koniec programu gdy wciśniemy klawisz Escape
	//{
	//	al_get_keyboard_state(&klawiatura);  // odczyt stanu klawiatury
	//	al_clear_to_color(al_map_rgb(0, 255, 0)); // wyczyszczenie aktualnego bufora ekranu
	//	al_draw_bitmap(obrazek, 0, 0, 0);  // wyświetlenie bitmapy "obrazek" na "Backbuffer" (bufor ekranu)
	//	al_flip_display(); // wyświetlenie aktualnego bufora na ekran
	//}
	//// usuwanie z pamięci okna, bitmap, audio, fontów ...itd.
	//al_destroy_display(okno);
	//al_destroy_bitmap(obrazek);

	al_init();

	ALLEGRO_DISPLAY *display = nullptr;
	al_set_app_name("Hello World from Allegro 5.1!");
	display = al_create_display(640, 480);
	if (display == nullptr)
	{
		std::cerr << "Well, something is not working..." << std::endl;
		al_rest(5.0);
		return EXIT_FAILURE;
	}

	al_clear_to_color(al_map_rgb(255, 255, 255));
	al_flip_display();
	al_rest(5.0);
	return 0;

	return 0;


	return 0;

}