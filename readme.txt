Image filtration made on CPU and GPU:
- To filtrate image, u need to have an .bmp file called image in the same folder as kernel.cu
- Application performs filtration on CPU or GPU
- User chooses which mask is used to filtrate image
- Other features contains: change bmp bit depth and change image to grayscale
- Filtered image will be created in the same directory as image.bmp
- filtered image name - "filtered_image.bmp"
- EasyBMP library was used to load and save images
- WinApi was used to make GUI

author:
Tomasz Pawłowski