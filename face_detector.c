#include <stdio.h>
#include <cv.h>
#include <highgui.h>

int main (int argc, char **argv) {
  int i;
  IplImage *src_img = 0, *src_gray = 0;
  const char *cascade_name = "haarcascade_frontalface_default.xml";
  CvHaarClassifierCascade *cascade = 0;
  CvMemStorage *storage = 0;
  CvSeq *faces;

  if (argc < 2 || (src_img = cvLoadImage (argv[1], CV_LOAD_IMAGE_COLOR)) == 0) return -1;
  src_gray = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_8U, 1);

  cascade = (CvHaarClassifierCascade *) cvLoad (cascade_name, 0, 0, 0);

  storage = cvCreateMemStorage (0);
  cvClearMemStorage (storage);
  cvCvtColor (src_img, src_gray, CV_BGR2GRAY);
  cvEqualizeHist (src_gray, src_gray);

  faces = cvHaarDetectObjects (src_gray, cascade, storage, 1.1, 4, CV_HAAR_SCALE_IMAGE, cvSize (40, 40));

  printf("[");
  for (i = 0; i < (faces ? faces->total : 0); i++) {
    CvRect *r = (CvRect *) cvGetSeqElem (faces, i);
    if (i != 0) printf(",");
    printf("{ \"x\": %d, \"y\": %d, \"width\": %d, \"height\": %d }", r->x, r->y, r->width, r->height);
  }
  printf("]");

  cvReleaseImage (&src_img);
  cvReleaseImage (&src_gray);
  cvReleaseMemStorage (&storage);

  return 0;
}

