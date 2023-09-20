#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <iomanip>


using namespace cv;
void manualBlur(const cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    // Define a blur kernel
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);

    // Apply the convolution operation to blur the input image
    cv::filter2D(inputImage, outputImage, -1, kernel);
}
uint8_t clampToByte(float value) {
    if (value < 0.0f) {
        return 0;
    } else if (value > 255.0f) {
        return 255;
    } else {
        return static_cast<uint8_t>(value);
    }
}
float cubicInterpolationWeight(float t) {
    // Cubic interpolation weight function
    if (t < 0) t = -t;
    float t2 = t * t;
    float t3 = t2 * t;
    if (t <= 1.0f) {
        return 1.5f * t3 - 2.5f * t2 + 1.0f;
    } else if (t <= 2.0f) {
        return -0.5f * t3 + 2.5f * t2 - 4.0f * t + 2.0f;
    } else {
        return 0.0f;
    }
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
cv::Mat resizeNearestNeighbor(const cv::Mat& src, int newWidth, int newHeight) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    cv::Mat dst(newHeight, newWidth, src.type());

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            // Calculate the corresponding coordinates in the source image
            int srcX = static_cast<int>(x * (srcWidth / static_cast<float>(newWidth)));
            int srcY = static_cast<int>(y * (srcHeight / static_cast<float>(newHeight)));

            // Nearest-neighbor interpolation
            dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(srcY, srcX);
        }
    }

    return dst;
}

cv::Mat resizeBilinear(const cv::Mat& src, int newWidth, int newHeight) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    cv::Mat dst(newHeight, newWidth, src.type());

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            // Calculate the corresponding coordinates in the source image
            double srcX = (x + 0.5) * (srcWidth / static_cast<float>(newWidth)) - 0.5;
            double srcY = (y + 0.5) * (srcHeight / static_cast<float>(newHeight)) - 0.5;

            int x0 = static_cast<int>(std::floor(srcX));
            int x1 = x0 + 1;
            int y0 = static_cast<int>(std::floor(srcY));
            int y1 = y0 + 1;

            x0 = std::max(0, std::min(x0, srcWidth - 1));
            x1 = std::max(0, std::min(x1, srcWidth - 1));
            y0 = std::max(0, std::min(y0, srcHeight - 1));
            y1 = std::max(0, std::min(y1, srcHeight - 1));

            // Bilinear interpolation
            double dx = srcX - x0;
            double dy = srcY - y0;

            cv::Vec3b pixel00 = src.at<cv::Vec3b>(y0, x0);
            cv::Vec3b pixel01 = src.at<cv::Vec3b>(y0, x1);
            cv::Vec3b pixel10 = src.at<cv::Vec3b>(y1, x0);
            cv::Vec3b pixel11 = src.at<cv::Vec3b>(y1, x1);

            cv::Vec3b interpolatedPixel;
            for (int c = 0; c < src.channels(); c++) {
                interpolatedPixel[c] = static_cast<uchar>(
                    (1 - dx) * (1 - dy) * pixel00[c] +
                    dx * (1 - dy) * pixel01[c] +
                    (1 - dx) * dy * pixel10[c] +
                    dx * dy * pixel11[c]
                );
            }

            dst.at<cv::Vec3b>(y, x) = interpolatedPixel;
        }
    }

    return dst;
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
float Triangular(float f) {
    f = f / 2.0;
    if (f < 0.0) {
        return (f + 1.0);
    } else {
        return (1.0 - f);
    }
}
float BellFunc( float x )
{
	float f = ( x / 2.0 ) * 1.5; // Converting -2 to +2 to -1.5 to +1.5
	if( f > -1.5 && f < -0.5 )
	{
		return( 0.5 * pow(f + 1.5, 2.0));
	}
	else if( f > -0.5 && f < 0.5 )
	{
		return 3.0 / 4.0 - ( f * f );
	}
	else if( ( f > 0.5 && f < 1.5 ) )
	{
		return( 0.5 * pow(f - 1.5, 2.0));
	}
	return 0.0;
}
float BSpline( float x )
{
	float f = x;
	if( f < 0.0 )
	{
		f = -f;
	}

	if( f >= 0.0 && f <= 1.0 )
	{
		return ( 2.0 / 3.0 ) + ( 0.5 ) * ( f* f * f ) - (f*f);
	}
	else if( f > 1.0 && f <= 2.0 )
	{
		return 1.0 / 6.0 * pow( ( 2.0 - f  ), 3.0 );
	}
	return 1.0;
}  
float CatMullRom( float x )
{
    const float B = 0.0;
    const float C = 0.5;
    float f = x;
    if( f < 0.0 )
    {
        f = -f;
    }
    if( f < 1.0 )
    {
        return ( ( 12 - 9 * B - 6 * C ) * ( f * f * f ) +
            ( -18 + 12 * B + 6 *C ) * ( f * f ) +
            ( 6 - 2 * B ) ) / 6.0;
    }
    else if( f >= 1.0 && f < 2.0 )
    {
        return ( ( -B - 6 * C ) * ( f * f * f )
            + ( 6 * B + 30 * C ) * ( f *f ) +
            ( - ( 12 * B ) - 48 * C  ) * f +
            8 * B + 24 * C)/ 6.0;
    }
    else
    {
        return 0.0;
    }
} 

cv::Vec3b BiCubicInterpolation(const cv::Mat& image, float x, float y) {
    //https://www.codeproject.com/Articles/236394/Bi-Cubic-and-Bi-Linear-Interpolation-with-GLSL
    // Adapted code
    int width = image.cols;
    int height = image.rows;
    
    int x_floor = static_cast<int>(floor(x));
    int y_floor = static_cast<int>(floor(y));
    
    cv::Vec3f nSum(0.0, 0.0, 0.0);
    cv::Vec3f nDenom(0.0, 0.0, 0.0);
    float a = x - x_floor;
    float b = y - y_floor;
  
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int xSample = x_floor + m;
            int ySample = y_floor + n;
            
            if (xSample >= 0 && xSample < width && ySample >= 0 && ySample < height) {

                //Gather image data and place it in vecdata.
                //As .at returns Vec3b we need to static cast to Vec3f
                cv::Vec3f vecData = static_cast<Vec3f>(image.at<cv::Vec3b>(ySample, xSample));
                float f = CatMullRom(static_cast<float>(m) - a);
                cv::Vec3f vecCooef1(f, f, f);
                float f1 = CatMullRom(-(static_cast<float>(n) - b));
                cv::Vec3f vecCoeef2(f1, f1, f1);
                //Clean up later. 
                nSum += vecData.mul(vecCoeef2.mul(vecCooef1));
                //nSum[0] += (vecData[0] * vecCoeef2[0] * vecCooef1)[0];
                //nSum[1] += (vecData[1] * vecCoeef2[1] * vecCooef1[1]);
                //nSum[2] += (vecData[2] * vecCoeef2[2] * vecCooef1[2]);
                nDenom += vecCoeef2.mul(vecCooef1);
                //nDenom[0] += (vecCoeef2[0] * vecCooef1[0]);
                //nDenom[1] += (vecCoeef2[1] * vecCooef1[1]);
                //nDenom[2] += (vecCoeef2[2] * vecCooef1[2]);
            }
        }
    } 
    //std::cout << "Sum: " << nSum<< std::endl;
    //std::cout << "Denom: " << nDenom<< std::endl;
    // OpenCV does not support element-wise division https://github.com/opencv/opencv/issues/23115
    // Have to brute force
    Vec3b output(0, 0, 0);
    for (int color = 0; color < image.channels(); color++)
    {
        //Calculations are done in floating point, so we need to clamp the answer to uchar
        output[color] = static_cast<uchar>(std::clamp((nSum[color] / nDenom[color]), 0.0f, 255.0f));

        //std::cout << "Ouutput: " << output[color] << std::endl;
        
    }
    // The output is a vector containing all the colorchannel BGR
    return output; 
}


cv::Mat resizeBicubic1(const cv::Mat& src, int newWidth, int newHeight) {
    cv::Mat dst(newHeight, newWidth, src.type());

    float scaleX = static_cast<float>(src.cols) / newWidth;
    float scaleY = static_cast<float>(src.rows) / newHeight;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;

            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));

            float dx = srcX - x0;
            float dy = srcY - y0;

            cv::Vec3f result(0.0f, 0.0f, 0.0f);

            for (int j = -1; j <= 2; j++) {
                for (int i = -1; i <= 2; i++) {
                    int xIdx = std::clamp(x0 + i, 0, src.cols - 1);
                    int yIdx = std::clamp(y0 + j, 0, src.rows - 1);

                    float weightX = cubicInterpolationWeight(dx - i);
                    float weightY = cubicInterpolationWeight(dy - j);

                    for (int c = 0; c < 3; c++) {
                        result[c] += src.at<cv::Vec3b>(yIdx, xIdx)[c] * weightX * weightY;
                    }
                }
            }

            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(std::clamp(result[c], 0.0f, 255.0f));
            }
        }
    }

    return dst;
}
//
/*
cv::Mat resizeBicubic(const cv::Mat& src, int newWidth, int newHeight) {
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

            struct pixel{
                float b;
                float g;
                float r;
            };   
            pixel splineY[4];
            cv::Vec3b interpolatedPixel;
            
                
            float vT = dx;
            float vTT = vT*vT;
            float vTTT = vTT*vT;

            float Q1 = 0.5f *(-vTTT + 2.0f * vTT -vT);
            float Q2 = 0.5f *(3.0f * vTTT - 5.0f*vTT + 2.0f);
            float Q3 = 0.5f *(-3.0f*vTTT + 4.0f*vTT +vT);
            float Q4 = 0.5f *(vTTT -vTT);
            for (int s = 0; s < 4; s++)
            {
                splineY[s].b = static_cast<float>(values[s][0][0]*Q1 + values[s][1][0]*Q2 + values[s][2][0]*Q3 + values[s][3][0]*Q4);
                splineY[s].g = static_cast<float>(values[s][0][1]*Q1 + values[s][1][1]*Q2 + values[s][2][1]*Q3 + values[s][3][1]*Q4);
                splineY[s].r = static_cast<float>(values[s][0][2]*Q1 + values[s][1][2]*Q2 + values[s][2][2]*Q3 + values[s][3][2]*Q4);
            }
            float vTy = dy;
            float vTTy = vTy*vTy;
            float vTTTy = vTTy*vTy;
            float Q1y = 0.5f *(-vTTTy + 2.0f * vTTy -vTy);
            float Q2y = 0.5f *(3.0f * vTTTy - 5.0f*vTTy + 2.0f);
            float Q3y = 0.5f *(-3.0f*vTTTy + 4.0f*vTTy +vTy);
            float Q4y = 0.5f *(vTTTy -vTTy);

            interpolatedPixel[0] = clampToByte(splineY[0].b*Q1y + splineY[1].b*Q2y + splineY[2].b*Q3y + splineY[3].b*Q4y);
            interpolatedPixel[1] = clampToByte(splineY[0].g*Q1y + splineY[1].g*Q2y + splineY[2].g*Q3y + splineY[3].g*Q4y);
            interpolatedPixel[2] = clampToByte(splineY[0].r*Q1y + splineY[1].r*Q2y + splineY[2].r*Q3y + splineY[3].r*Q4y);
            
            dst.at<cv::Vec3b>(y, x) = interpolatedPixel;
        }
    }   

    return dst;
}
*/
cv::Mat resizeBicubic2(const cv::Mat& src, int newWidth, int newHeight) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;
    float scaleX = static_cast<float>(src.cols) / newWidth;
    float scaleY = static_cast<float>(src.rows) / newHeight;
    cv::Mat dst(newHeight, newWidth, src.type());

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            // Calculate the corresponding coordinates in the source image
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            dst.at<cv::Vec3b>(y, x) = BiCubicInterpolation(src, srcX, srcY);;
        }
    }   

    return dst;
}


/*
cv::Mat resizeBicubicWithKernel(const cv::Mat& src, int newWidth, int newHeight) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    cv::Mat dst(newHeight, newWidth, src.type());

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            // Calculate the corresponding coordinates in the source image
            float srcX = x * (srcWidth / static_cast<float>(newWidth));
            float srcY = y * (srcHeight / static_cast<float>(newHeight));

            int x0 = static_cast<int>(std::floor(srcX));
            int x1 = x0 + 1;
            int x2 = x0 + 2;
            int x3 = x0 + 3;

            int y0 = static_cast<int>(std::floor(srcY));
            int y1 = y0 + 1;
            int y2 = y0 + 2;
            int y3 = y0 + 3;

            x0 = std::max(0, std::min(x0, srcWidth - 1));
            x1 = std::max(0, std::min(x1, srcWidth - 1));
            x2 = std::max(0, std::min(x2, srcWidth - 1));
            x3 = std::max(0, std::min(x3, srcWidth - 1));

            y0 = std::max(0, std::min(y0, srcHeight - 1));
            y1 = std::max(0, std::min(y1, srcHeight - 1));
            y2 = std::max(0, std::min(y2, srcHeight - 1));
            y3 = std::max(0, std::min(y3, srcHeight - 1));

            // Bicubic interpolation using a kernel
            float dx = srcX - x0;
            float dy = srcY - y0;

            cv::Vec3b interpolatedPixel(0, 0, 0);

            // Bicubic kernel values
            float kernel[4][4] = {
                {1.0f, 2.0f, 0.0f, 0.0f},
                {-0.5f, 0.0f, 0.5f, 0.0f},
                {0.0f, -1.0f, 1.0f, 0.0f},
                {0.0f, 2.0f, -5.0f, 3.0f}
            };

            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    int px = x0 + i;
                    int py = y0 + j;

                    cv::Vec3b pixel = src.at<cv::Vec3b>(py, px);

                    for (int c = 0; c < src.channels(); c++) {
                        interpolatedPixel[c] += pixel[c] * kernel[j][i];
                    }
                }
            }

            for (int c = 0; c < src.channels(); c++) {
                interpolatedPixel[c] = std::max(0, std::min(255, static_cast<int>(interpolatedPixel[c])));
            }

            dst.at<cv::Vec3b>(y, x) = interpolatedPixel;
        }
    }

    return dst;
}*/
int main(int argc, char** argv )
{

    cv::Mat img;
    img = imread("C:/Users/William Haugerud/Downloads/lenna.png");
    //std::cout << "IMG type " << img.type() << std::endl;
    if ( !img.data )
    {
        printf("No image data \n");
        return -1;
    }
    //printf("Num Rows \n");
    
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    
    //waitKey(0);

    int numRows = img.rows;
    int numCols = img.cols;
    float scale_factor = 1.5;

    // Print the number of rows and columns
    std::cout << "Number of rows: " << numRows << std::endl;
    std::cout << "Number of columns: " << numCols << std::endl;
    cv::Mat scaled_img_bilinear;
    //scaleImage(img, scaled_img, 0.5);
    scaled_img_bilinear = resizeBilinear(img, int(img.rows * scale_factor), int(img.cols * scale_factor));
    cv::Mat scaled_img_bicubic;
    //scaleImage(img, scaled_img, 0.5);
    scaled_img_bicubic = resizeBicubic2(img, int(img.rows * scale_factor), int(img.cols * scale_factor));
    //resize(img, scaled_img_bicubic, cv::Size(int(img.rows * scale_factor), int(img.cols * scale_factor)), INTER_CUBIC);
    //namedWindow("Scaled image", WINDOW_AUTOSIZE);
    cv::Mat scaled_img_point;
    //blurImage(img, blurred_img, 5);
    scaled_img_point = resizeNearestNeighbor(img, int(img.rows * scale_factor), int(img.cols * scale_factor));
    //namedWindow("Blurred image", WINDOW_AUTOSIZE);
    cv::namedWindow("Display image", WINDOW_NORMAL);
    cv::namedWindow("Bilinear", WINDOW_NORMAL);
    cv::namedWindow("Bicubic", WINDOW_NORMAL);
    cv::namedWindow("Point", WINDOW_NORMAL);
    cv::resizeWindow("Display image", 1024, 1024);
    cv::resizeWindow("Bilinear", 1024, 1024);
    cv::resizeWindow("Bicubic", 1024, 1024);
    cv::resizeWindow("Point", 1024, 1024);

    
    cv::imshow("Display image", img);
    cv::imshow("Bilinear", scaled_img_bilinear);
    cv::imshow("Bicubic", scaled_img_bicubic);
    cv::imshow("Point", scaled_img_point);
    waitKey(0);
    cv::destroyAllWindows();
    return 0;
}