#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

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
