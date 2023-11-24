#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O



const int array_size = 16;

double triangular(float f) {
    f = f / 2.0;
    if (f < 0.0) {
        return (f + 1.0);
    } else {
        return (1.0 - f);
    }
}
double bellFunc( double x )
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
double BSpline( float x )
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
double catMullRom(float x)
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
//https://www.codeproject.com/Articles/236394/Bi-Cubic-and-Bi-Linear-Interpolation-with-GLSL
cv::Vec3b biCubicInterpolation(const cv::Mat& image, float x, float y) {
    int width = image.cols;
    int height = image.rows;

    int x_floor = static_cast<int>(floor(x));
    int y_floor = static_cast<int>(floor(y));

    cv::Vec3f n_sum(0.0, 0.0, 0.0);
    cv::Vec3f n_denom(0.0, 0.0, 0.0);
    float a = x - x_floor;
    float b = y - y_floor;

    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int x_sample = x_floor + m;
            int y_sample = y_floor + n;
            // Handle edge cases.
            if (x_sample >= 0 && x_sample < width && y_sample >= 0 && y_sample < height) {
                // Gather image data and place it in vecdata.
                // As .at returns Vec3b we need to static cast to Vec3f
                // Change to Vec3d to use double instead for the return?
                cv::Vec3f vec_data = static_cast<cv::Vec3f>(image.at<cv::Vec3b>(y_sample, x_sample));
                float f = catMullRom(static_cast<float>(m) - a);
                cv::Vec3f vec_coeff1(f, f, f);
                float f1 = catMullRom(-(static_cast<float>(n) - b));
                cv::Vec3f vec_coeff2(f1, f1, f1);
                n_sum += vec_data.mul(vec_coeff2.mul(vec_coeff1));
                n_denom += vec_coeff2.mul(vec_coeff1);
            }
        }
    }
    // OpenCV does not support element-wise division https://github.com/opencv/opencv/issues/23115
    // Have to brute force
    cv::Vec3b output(0, 0, 0);
    for (int color = 0; color < image.channels(); color++) {
        // Calculations are done in floating point, so we need to clamp the answer to uchar
        output[color] = static_cast<uchar>(std::clamp((n_sum[color] / n_denom[color]), 0.0f, 255.0f));
    }
    // The output is a vector containing all the color channel BGR
    return output;
}


int* generateRandomArray(int size) {
    // Seed the random number generator with the current time
    srand(static_cast<unsigned>(time(0)));
    int count = 0;
    // Create a dynamic array of the specified size
    int* my_array = new int[size];

    // Initialize the array with zeros
    for (int i = 0; i < size; i++) {
        my_array[i] = 1;
    }
    //random unobtainable number
    int last_placed = 300;
    for (int i = 0; i < count; i++)
    {   
        before_loop:
        int random_index = rand() % size;
        if (my_array[random_index] != 0){
            my_array[random_index] = 0;
            
        }else {
            goto before_loop;
        }   
    }

    return my_array;

}

cv::Vec3b BiCubicInterpolationStochastic(const cv::Mat& image, float x, float y) {
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

    int* result_array_b = generateRandomArray(array_size);
    int* result_array_g = generateRandomArray(array_size);
    int* result_array_r = generateRandomArray(array_size);
  
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int xSample = x_floor + m;
            int ySample = y_floor + n;
            // Handle edge cases. 


            if (xSample >= 0 && xSample < width && ySample >= 0 && ySample < height && result_array_b[(m + 1) + (n + 1)] == 1) {

            //Gather image data and place it in vecdata.
            //As .at returns Vec3b we need to static cast to Vec3f
            cv::Vec3f vecData = static_cast<cv::Vec3f>(image.at<cv::Vec3b>(ySample, xSample));
            float f = catMullRom(static_cast<float>(m) - a);
            cv::Vec3f vecCooef1(f, f, f);
            float f1 = catMullRom(-(static_cast<float>(n) - b));
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
    cv::Vec3b output(0, 0, 0);
    for (int color = 0; color < image.channels(); color++)
    {
        //Calculations are done in floating point, so we need to clamp the answer to uchar
        output[color] = static_cast<uchar>(std::clamp((nSum[color] / nDenom[color]), 0.0f, 255.0f));

        //std::cout << "Ouutput: " << output[color] << std::endl;
        
    }
    // The output is a vector containing all the colorchannel BGR
    delete[] result_array_b;
    delete[] result_array_g;
    delete[] result_array_r;
    return output; 
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
cv::Mat resizeBicubic(const cv::Mat& src, int new_width, int new_height) {
    int src_width = src.cols;
    int src_height = src.rows;
    float scale_x = static_cast<float>(src.cols) / new_width;
    float scale_y = static_cast<float>(src.rows) / new_height;
    cv::Mat dst(new_height, new_width, src.type());

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // Calculate the corresponding coordinates in the source image
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            dst.at<cv::Vec3b>(y, x) = biCubicInterpolation(src, src_x, src_y);
        }
    }   

    return dst;
}

