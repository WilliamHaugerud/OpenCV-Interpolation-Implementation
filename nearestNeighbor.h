#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

cv::Mat resizeNearestNeighbor(const cv::Mat& src, int new_width, int new_height) {
    int src_width = src.cols;
    int src_height = src.rows;

    cv::Mat dst(new_height, new_width, src.type());

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // Calculate the corresponding coordinates in the source image
            int src_x = static_cast<int>(x * (src_width / static_cast<float>(new_width)));
            int src_y = static_cast<int>(y * (src_height / static_cast<float>(new_height)));

            // Nearest-neighbor interpolation
            dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(src_y, src_x);
        }
    }

    return dst;
}