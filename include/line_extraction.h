#ifndef LINE_EXTRACTION_H
#define LINE_EXTRACTION_H

#include <Eigen/Eigen>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <cmath>

using namespace std;

namespace line_based_plane_segment
{

struct ScanPoint
{
    float x, y;
    float phi, rho;
    float stdrho;
    float cosPhi, sinPhi;
};

struct ScanLine
{
    ScanLine() : alpha(0), r(0), saa(0), srr(0), sar(0), begin(0), end(0), size(0){}
    ScanLine(float alpha_, float r_) : alpha(alpha_), r(r_), saa(0), srr(0), sar(0), begin(0), end(0), size(0) {}

    float alpha, r;
    float saa, srr, sar;
    int begin, end, size;
    float length;
    float length_sigma2;
};

struct ScanSegment
{
    ScanSegment():begin(0), end(0), size(0){}
    ScanSegment(int _begin,int _end):begin(_begin),end(_end),size(end-begin+1){}

    int begin;
    int end;
    int size;
};

typedef std::vector<ScanLine> VectorScanLines;
typedef std::vector<ScanPoint> VectorScanPoints;
typedef std::vector<ScanSegment> VectorScanSegments;

class LineRegressionSegmentation{
public:
    LineRegressionSegmentation();
    LineRegressionSegmentation(int window_size, float fidelity, int min_inlier);
    void setParameters(int window_size, float fidelity, int min_inlier);
    void segment(VectorScanPoints &scan_point, VectorScanSegments &scan_segment);

private:
    void slideWindowFitLines(VectorScanPoints &scan_point, VectorScanLines &lines);
    bool fitLinePolar(VectorScanPoints &scan, ScanLine &line);
    float calcCompactness(VectorScanLines &lines);
    void findLineRegions(vector<float> &compactness, float threshfidel, VectorScanSegments &indices);
    void refineRegion(VectorScanSegments &indices);
    void meanwm(vector<Eigen::Vector2f> xv, vector<Eigen::Matrix2f> Cv,
                Eigen::Vector2f &xw, Eigen::Matrix2f &Cw);
    float mahalanobisar(ScanLine l1, ScanLine l2);

public:
    // line extraction parameters
    int window_size_;        // window size in points
    double thresh_fidelity_;  // model fidelity threshold
    int min_inlier_;
};

}

#endif // LINE_EXTRACTION_H
