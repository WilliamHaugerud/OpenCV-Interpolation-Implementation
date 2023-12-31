#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <time.h>

#include "nearestNeighbor.h"
#include "bilinear.h"
#include "bicubic.h"
#include "imageComparison.h"


using namespace cv;
void manualBlur(const cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    // Define a blur kernel
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);

    // Apply the convolution operation to blur the input image
    cv::filter2D(inputImage, outputImage, -1, kernel);
}
//Average quadrants and bilinear interpolate
cv::Mat resizeCustom(const cv::Mat& src, int newWidth, int newHeight) {

    int srcWidth = src.cols;
    int srcHeight = src.rows;
    int padding = 4;
    int padding2 = padding / 2;
        // Print the number of rows and columns
    std::cout << "Number of rows: " << srcWidth << std::endl;
    std::cout << "Number of columns: " << srcHeight << std::endl;
    cv::Mat dst(newHeight, newWidth, src.type());
    //cv::Mat dst = cv::Mat::zeros(newHeight + padding, newWidth + padding, src.type());
    //src.copyTo(dst(cv::Rect(padding2, padding2, srcWidth, srcHeight)));
    std::cout << "Number of rows: " << dst.cols << std::endl;
    std::cout << "Number of columns: " << dst.rows << std::endl;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            // Calculate the corresponding coordinates in the source image
            float srcX = x * (srcWidth / static_cast<float>(newWidth));
            float srcY = y * (srcHeight / static_cast<float>(newHeight));

            int x0 = static_cast<int>(std::floor(srcX));
            //int x1 = x0 + 1;
            //int x2 = x0 + 2;
            //int x3 = x0 + 3;

            int y0 = static_cast<int>(std::floor(srcY));
            //int y1 = y0 + 1;
            //int y2 = y0 + 2;
            //int y3 = y0 + 3;

            x0 = std::max(0, std::min(x0, srcWidth - 1));
            //x1 = std::max(0, std::min(x1, srcWidth - 1));
            //x2 = std::max(0, std::min(x2, srcWidth - 1));
            //x3 = std::max(0, std::min(x3, srcWidth - 1));

            y0 = std::max(0, std::min(y0, srcHeight - 1));
            //y1 = std::max(0, std::min(y1, srcHeight - 1));
            //y2 = std::max(0, std::min(y2, srcHeight - 1));
            //y3 = std::max(0, std::min(y3, srcHeight - 1));



            // Bicubic interpolation
            // Finding offset
            float dx = srcX - x0;
            float dy = srcY - y0;

            cv::Vec3b values[4][4];
            for (int j = -1; j < 3; j++) {
                    for (int i = -1; i < 3; i++) {
                        // Handle border cases. When outside of picture use closest pixel value.
                        // This duplicates the outside values.
                        if ((y0 + j < 0) || (y0 + j > srcHeight - 1)  && (x0 + i < 0) || (x0 + i > srcWidth - 1)){
                            values[i+1][j+1] = src.at<cv::Vec3b>(y0, x0);
                        }
                        else if ((y0 + j < 0) || (y0 + j > srcHeight - 1))
                        {
                            values[i+1][j+1] = src.at<cv::Vec3b>(y0, x0 + i);
                        }
                        else if ((x0 + i < 0) || (x0 + i > srcWidth - 1))
                        {
                            values[i+1][j+1] = src.at<cv::Vec3b>(y0 + j, x0);
                        }
                        else{
                            values[i+1][j+1] = src.at<cv::Vec3b>(y0 + j, x0 + i);
                        }
                        
                    }
                }

            cv::Vec3b interpolatedPixel;
            for (int c = 0; c < src.channels(); c++) {

                float quadrantA = 0.0f;
                float quadrantB = 0.0f;
                float quadrantC = 0.0f;
                float quadrantD = 0.0f;

                // Calculate the average of each quadrant
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                    quadrantA += values[i][j][c];
                    quadrantB += values[i + 2][j][c];
                    quadrantC += values[i][j + 2][c];
                    quadrantD += values[i + 2][j + 2][c];
                    }
                }
                int numPixelsInQuadrant = 4;
                quadrantA /= numPixelsInQuadrant;
                quadrantB /= numPixelsInQuadrant;
                quadrantC /= numPixelsInQuadrant;
                quadrantD /= numPixelsInQuadrant;

                interpolatedPixel[c] = static_cast<uchar>(
                    (1 - dx) * (1 - dy) * quadrantA +
                    dx * (1 - dy) * quadrantB +
                    (1 - dx) * dy * quadrantC +
                    dx * dy * quadrantD
                );

            }

            dst.at<cv::Vec3b>(y, x) = interpolatedPixel;
        }
    }   

    return dst;
}

void blurImage(cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    // Apply Gaussian blur to the input image
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), 0);
}
void scaleImage(cv::Mat& input_img, cv::Mat& output_img, float scale) {
    // Scale image. Using int will scale to nearest whole number
    int target_width = static_cast<int>(input_img.rows * scale);
    int target_height = static_cast<int>(input_img.cols * scale);
    /*INTER_NEAREST - a nearest-neighbor interpolation
    INTER_LINEAR - a bilinear interpolation (used by default)
    INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood*/
    resize(input_img, output_img, cv::Size(target_width, target_height), INTER_CUBIC);
}

cv::Mat stochasticExample (int width, int height, int size){
    // Create a distribution of jittered samples.
    Mat image(width, height, CV_8UC3, Scalar(0, 0, 0));
    std::srand(static_cast<unsigned>(time(0)));
    int x,y;
    int jx,jy;
    for (y = 0; y < height; y+= size){
        for (x = 0; x < width; x += size){
            double random_number_x = static_cast<double>(std::rand()) / RAND_MAX;
            double random_number_y = static_cast<double>(std::rand()) / RAND_MAX;
            jx = static_cast<int>(size*random_number_x);
            jy = static_cast<int>(size*random_number_y);
            image.at<cv::Vec3b>(y+jy, x+jx) = cv::Vec3b(255, 255, 255); // White color
        }
    }
    return image;
}


/*
int main(int argc, char** argv )
{
    int width = 512;
    int height = 512;
    cv::Mat img;
    img = imread("C:/Users/William Haugerud/Downloads/lenna.png");
    //std::cout << "IMG type " << img.type() << std::endl;
    if ( !img.data )
    {
        printf("No image data \n");
        return -1;
    }

    int* result_array = generateRandomArray(array_size);
    for (int i = 0; i < array_size; i++) {
        std::cout << result_array[i] << " ";
    }
    cv::Mat stoch_img = stochasticExample(width, height, 4);
    cv::Mat scaled_img; 
    cv::Mat scaled_img_bicubic;
    float scale_factor = 2.0;
    cv::resize(img, scaled_img, cv::Size(img.rows/2, img.cols/2), INTER_NEAREST);
    int scaled_rows = scaled_img.rows * scale_factor;
    int scaled_cols = scaled_img.cols * scale_factor;
    scaled_img_bicubic = resizeBicubic2(scaled_img, int(scaled_rows), int(scaled_cols));
    

    cv::namedWindow("Stochastic example", WINDOW_NORMAL);
    cv::namedWindow("Bicubic", WINDOW_NORMAL);
    cv::resizeWindow("Stochastic example", 512, 512);
    //cv::resizeWindow("Bicubic", 1024, 1024);
    cv::imshow("Stochastic example", stoch_img);
    cv::imshow("Bicubic", scaled_img_bicubic);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete[] result_array;
    return 0;
}*/

int main(int argc, char** argv ){

    cv::Mat img_original;
    img_original = imread("C:/Users/William Haugerud/Downloads/Green_sea_shell_1024x768.jpg");
    //std::cout << "IMG type " << img_original.type() << std::endl;
    if ( !img_original.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::Mat img_small;
    img_small = imread("C:/Users/William Haugerud/Downloads/Green_sea_shell_320x240.jpg");
    //std::cout << "IMG type " << img_small.type() << std::endl;
    if ( !img_small.data )
    {
        printf("No image data \n");
        return -1;
    }

    int num_cols_original = img_original.rows;
    int num_rows_original = img_original.cols;
    

    int num_rows_small = img_small.rows;
    int num_cols_small = img_small.cols;


    //Adjust the scale factor
    float scale_factor = 2.0;
    // Print the number of rows and columns
    std::cout << "Number of rows: " << num_rows_small << std::endl;
    std::cout << "Number of columns: " << num_cols_small << std::endl;
    //Resizing image with bilinear
    cv::Mat scaled_img_bilinear;
    scaled_img_bilinear = resizeBilinear(img_small, int(num_rows_original), int(num_cols_original));
    //Resizing with opencv bilinear
    cv::Mat scaled_img_bilinear_opencv;
    resize(img_small, scaled_img_bilinear_opencv, cv::Size(int(num_rows_original), int(num_cols_original)), 0, 0, INTER_LINEAR);
    //Resizing image with bicubic
    cv::Mat scaled_img_bicubic;
    scaled_img_bicubic = resizeBicubic(img_small, int(num_rows_original), int(num_cols_original));
    //Resizing image with opencv bicubic
    cv::Mat scaled_img_bicubic_opencv;
    resize(img_small, scaled_img_bicubic_opencv, cv::Size(int(num_rows_original), int(num_cols_original)), 0, 0, INTER_CUBIC);
    //Resizing image with nearest neighbor
    cv::Mat scaled_img_nearest;
    scaled_img_nearest = resizeNearestNeighbor(img_small, int(num_rows_original), int(num_cols_original));
    cv::Mat scaled_img_nearest_opencv;
    resize(img_small, scaled_img_nearest_opencv, cv::Size(int(num_rows_original), int(num_cols_original)), 0, 0, INTER_NEAREST);

    //Showing images and resizing the window to see better
    //cv::namedWindow("Original", WINDOW_NORMAL);
    //cv::namedWindow("Bilinear", WINDOW_NORMAL);
    //cv::namedWindow("Bicubic", WINDOW_NORMAL);
    //cv::namedWindow("Bicubic opencv", WINDOW_NORMAL);
    //cv::namedWindow("Nearest", WINDOW_NORMAL);
 
    //cv::resizeWindow("Original", 1024, 768);
    //cv::resizeWindow("Bilinear", 1024, 768);
    //cv::resizeWindow("Bicubic", 1024, 768);
    //cv::resizeWindow("Bicubic opencv", 1024, 768);
    //cv::resizeWindow("Nearest", 1024, 768);

    
    std::cout << "Number of rows opencv: " << scaled_img_bicubic_opencv.rows << std::endl;
    std::cout << "Number of columns opencv: " << scaled_img_bicubic_opencv.cols << std::endl;

    std::cout << "Number of rows mine: " << scaled_img_bicubic.rows << std::endl;
    std::cout << "Number of columns mine: " << scaled_img_bicubic.cols << std::endl;
    
    //MSSIM/PSNR bilinear 
    std::cout << "SSIM bil mine: " << getMSSIM(img_original, scaled_img_bilinear) << std::endl;
    std::cout << "PSNR bil mine: " << getPSNR(img_original, scaled_img_bilinear) << std::endl;
    std::cout << "SSIM bil opencv: " << getMSSIM(scaled_img_bilinear_opencv, img_original) << std::endl;
    std::cout << "PSNR bil opencv: " << getPSNR(scaled_img_bilinear_opencv, img_original) << std::endl;
    //MSSIM/PSNR bicubic
    std::cout << "SSIM bicubic mine : " << getMSSIM(img_original, scaled_img_bicubic) << std::endl;
    std::cout << "PSNR bicubic mine : " << getPSNR(img_original, scaled_img_bicubic) << std::endl;
    std::cout << "SSIM bicubic opencv : " << getMSSIM(img_original, scaled_img_bicubic_opencv) << std::endl;
    std::cout << "PSNR bicubic opencv : " << getPSNR(img_original, scaled_img_bicubic_opencv) << std::endl;
    //MSSIM/PSNR nearest
    std::cout << "SSIM nearest mine: " << getMSSIM(img_original, scaled_img_nearest) << std::endl;
    std::cout << "PSNR nearest mine: " << getPSNR(img_original, scaled_img_nearest) << std::endl;
    std::cout << "SSIM nearest opencv: " << getMSSIM(img_original, scaled_img_nearest_opencv) << std::endl;
    std::cout << "PSNR nearest opencv: " << getPSNR(img_original, scaled_img_nearest_opencv) << std::endl;
 
    
    cv::imshow("Original", img_original);
    cv::imshow("Bilinear", scaled_img_bilinear);
    cv::imshow("Bilinear opencv", scaled_img_nearest_opencv);
    cv::imshow("Bicubic", scaled_img_bicubic);
    cv::imshow("Bicubic opencv", scaled_img_bicubic_opencv);
    cv::imshow("Nearest opencv", scaled_img_nearest_opencv);
    cv::imshow("Nearest", scaled_img_nearest);
    waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
