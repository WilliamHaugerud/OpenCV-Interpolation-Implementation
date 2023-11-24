#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

cv::Mat resizeBilinear(const cv::Mat& src, int new_width, int new_height) {
    int src_width = src.cols;
    int src_height = src.rows;

    cv::Mat dst(new_height, new_width, src.type());

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // Calculate the corresponding coordinates in the source image
            double src_x = (x + 0.5) * (src_width / static_cast<float>(new_width)) - 0.5;
            double src_y = (y + 0.5) * (src_height / static_cast<float>(new_height)) - 0.5;

            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = x0 + 1;
            int y0 = static_cast<int>(std::floor(src_y));
            int y1 = y0 + 1;

            x0 = std::max(0, std::min(x0, src_width - 1));
            x1 = std::max(0, std::min(x1, src_width - 1));
            y0 = std::max(0, std::min(y0, src_height - 1));
            y1 = std::max(0, std::min(y1, src_height - 1));

            // Bilinear interpolation
            double dx = src_x - x0;
            double dy = src_y - y0;

            cv::Vec3b pixel00 = src.at<cv::Vec3b>(y0, x0);
            cv::Vec3b pixel01 = src.at<cv::Vec3b>(y0, x1);
            cv::Vec3b pixel10 = src.at<cv::Vec3b>(y1, x0);
            cv::Vec3b pixel11 = src.at<cv::Vec3b>(y1, x1);

            cv::Vec3b interpolated_pixel;
            for (int c = 0; c < src.channels(); c++) {
                interpolated_pixel[c] = static_cast<uchar>(
                    (1 - dx) * (1 - dy) * pixel00[c] +
                    dx * (1 - dy) * pixel01[c] +
                    (1 - dx) * dy * pixel10[c] +
                    dx * dy * pixel11[c]
                );
            }

            dst.at<cv::Vec3b>(y, x) = interpolated_pixel;
        }
    }

    return dst;
}
