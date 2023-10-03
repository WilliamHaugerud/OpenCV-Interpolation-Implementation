#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

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