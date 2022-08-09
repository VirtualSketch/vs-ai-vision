#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv4/opencv2/core/matx.hpp>
#include <string>

using namespace std;
using namespace cv;

void showImage(string name, Mat image) {
    imshow(name, image);
    waitKey(0);
    destroyWindow(name);
}

Mat deskew(Mat &img) {
    Moments m = moments(img);

    if (abs(m.mu02) < 1e-2)
        return img.clone();

    float skew = m.mu11 / m.mu02;

    Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * 20 * skew, 0, 1, 0);

    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());

    warpAffine(img, imgOut, warpMat, imgOut.size(), WARP_INVERSE_MAP | INTER_LINEAR);

    return imgOut;
}

int main() {

    Mat image = imread("assets/mynumbers.jpeg");
    /* bitwise_not(image, image); */
    /* showImage("Original", image); */

    Mat resizedImage;
    resize(image, resizedImage, Size(1000, 250));
    /* showImage("Resized", resizedImage); */

    Mat grayImage;
    cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);
    /* showImage("Grayscale", grayImage); */

    Mat threshImage;
    threshold(grayImage, threshImage, 128, 255, THRESH_BINARY);
    /* showImage("Thresh", threshImage); */

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(threshImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    Mat imageClone = resizedImage.clone();
    drawContours(imageClone, contours, -1, Scalar(0, 255, 0), 1);
    /* showImage("Contours with NONE", imageClone); */

    imageClone = resizedImage.clone();
    vector<Mat> numbers;
    vector<int> numbersX, numbersY;
    
    for (int i = 0; i < contours.size(); i++) {
        Rect bound = boundingRect(contours.at(i));
        if (bound.area() > 500 && bound.area() < 1000 * 250) {
            cout << bound.area() << endl;

            int beginX = bound.x - 10;
            int endX = bound.x + bound.width + 10;
            int beginY = bound.y - 10;
            int endY = bound.y + bound.height + 10;

            numbersX.push_back(bound.x);
            numbersY.push_back(bound.y);

            numbers.push_back(threshImage.colRange(beginX, endX).rowRange(beginY, endY).clone());

            rectangle(imageClone, Point(beginX, beginY), Point(endX, endY), Scalar(0, 255, 0), 2);
        }
    }

    showImage("Bounded", imageClone);

    int quantNumbers = numbersX.size();

    /* cout << quantNumbers << " " << numbersY.size() << " " << numbers.size() << endl; */

    for (int j = quantNumbers - 1; j > 0; j--) {
        for (int i = 0; i < j; i++) {
            bool changed = false;

            if (numbersX.at(i) > numbersX.at(i + 1)) {
                int temp = numbersX.at(i);
                numbersX.at(i) = numbersX.at(i + 1);
                numbersX.at(i + 1) = temp;
                changed = true;
            }
            /* if (numbersY.at(i) > numbersY.at(i + 1)) { */
            /*     int temp = numbersY.at(i); */
            /*     numbersY.at(i) = numbersY.at(i + 1); */
            /*     numbersY.at(i + 1) = temp; */
            /*     changed = true; */
            /* } */
            if (changed) {
                Mat temp = numbers.at(i);
                numbers.at(i) = numbers.at(i + 1);
                numbers.at(i + 1) = temp;
            }
        }
    }

    HOGDescriptor hog = HOGDescriptor(
        /* Size(20, 20), // winSize */
        /* Size(8, 8), // blockSize */
        /* Size(4, 4), // blockStride */
        /* Size(8, 8), // cellSize */
        Size(40, 40), // winSize
        Size(16, 16), // blockSize
        Size(8, 8), // blockStride
        Size(16, 16), // cellSize
        9, // nbins
        1, // derivAper
        -1.0, // winSigma
        HOGDescriptor::L2Hys, // histogramNormType
        0.2, // L2HysThresh
        false, // gammal correction
        64, // nlevels = 64
        true // use signed gradients
    );

    vector<vector<float>> hogDescriptors;

    for (int i = 0; i < quantNumbers; i++) {
        /* cout << x << endl; */
        Mat currentNum = numbers.at(i);
        Mat resizedNum;

        string textNum = to_string(numbersY.at(i));

        resize(currentNum, resizedNum, Size(40, 40));
        
        Mat deskewedNum = deskew(resizedNum);

        vector<float> numDescriptors;
        hog.compute(deskewedNum, numDescriptors);

        /* cout << numDescriptors[0] << " " << numDescriptors[1] << " " << numDescriptors[2] << endl; */
        /* showImage("Image", deskewedNum); */

        hogDescriptors.push_back(numDescriptors);
    }

    /* for (int i = 0; i < trainLabels.size(); i++) { */
    /*     if (i == trainLabels.size() - 1) */
    /*         cout << trainLabels.at(i) << endl; */
    /*     else */
    /*         cout << trainLabels.at(i) << ", "; */
    /* } */

    int descriptorSize = hogDescriptors.at(0).size();

    Mat hogMat(hogDescriptors.size(), descriptorSize, CV_32FC1);

    for (int i = 0; i < hogDescriptors.size(); i++) {
        for (int j = 0; j < descriptorSize; j++) {
            hogMat.at<float>(i, j) = hogDescriptors[i][j];
        }
    }

    Ptr<ml::SVM> svm = ml::SVM::load("assets/trainedData.yml");

    svm -> setC(12.5);
    svm -> setGamma(0.5);
    /* svm -> setKernel(ml::SVM::RBF); */
    /* svm -> setType(ml::SVM::C_SVC); */

    Mat answerMat;

    // fire time → TESTING
    svm -> predict(hogMat, answerMat);

    for (int i = 0; i < answerMat.rows; i++) {
        /* imshow("Predicted number: " + to_string(answerMat.at<float>(i, 0)), numbers.at(numbersX.at(i))); */
        /* waitKey(0); */
        /* destroyAllWindows(); */
        /* cout << answerMat.at<float>(i, 0) << endl; */
        if (i == 0)
            putText(imageClone, to_string((int) answerMat.at<float>(i, 0)), Point(numbersX.at(i) - 20, numbersY.at(i) - 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255));
        else if (i == answerMat.rows - 1)
            putText(imageClone, to_string((int) answerMat.at<float>(i, 0)), Point(numbersX.at(i), numbersY.at(i)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255));
        else
            putText(imageClone, to_string((int) answerMat.at<float>(i, 0)), Point(numbersX.at(i) - 20, numbersY.at(i) - 10), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255));
        
    }

    showImage("Final", imageClone);

    return 0;
}
