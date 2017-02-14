#include "line_extraction.h"

namespace line_based_plane_segment
{

LineRegressionSegmentation::LineRegressionSegmentation()
    : window_size_(11)
    , thresh_fidelity_(0.2)
    , min_inlier_(23)
{}

LineRegressionSegmentation::LineRegressionSegmentation(int window_size, float fidelity, int min_inlier)
    : window_size_(window_size)
    , thresh_fidelity_(fidelity)
    , min_inlier_(min_inlier)
{}


void LineRegressionSegmentation::setParameters(int window_size, float fidelity, int min_inlier)
{
    window_size_ = window_size;
    thresh_fidelity_ = fidelity;
    min_inlier_ = min_inlier;
}

void LineRegressionSegmentation::segment(VectorScanPoints &scan_point, VectorScanSegments &scan_segment)
{
    VectorScanLines lines;
    slideWindowFitLines(scan_point, lines);
//        // output scanpoint
//        cout << " - points: " << scan_point.size() << endl;
//        for(int i = 0; i < scan_point.size(); i++)
//        {
//            cout << " " << scan_point[i].stdrho;
//        }
//        cout << endl;
//        // output lines
//        cout << " - lines: " << lines.size() << endl;
//        for(int i = 0; i < lines.size(); i++)
//        {
//            cout << " (" << lines[i].alpha << ", " << lines[i].r << ")";
//        }
//        cout << endl;

    // Calculate Model Fidelity (=COmpactness) Measure --> compactness
    vector<float> compactness;  //
    for(int i = 1; i < (lines.size()-1); i++)
    {
        VectorScanLines ls(3);
        ls[0] = lines[i-1];
        ls[1] = lines[i];
        ls[2] = lines[i+1];
        compactness.push_back(calcCompactness(ls));
    }

//        cout << " - compactness: " << compactness.size() << endl;
//        for(int i = 0; i < lines.size(); i++)
//        {
//            cout << " " << compactness[i];
//        }
//        cout << endl;

    // Apply Model Fidelity Threshold --> rawSegs(1/2/3, :)
    findLineRegions(compactness, thresh_fidelity_, scan_segment);
    refineRegion(scan_segment);
}

void LineRegressionSegmentation::slideWindowFitLines(VectorScanPoints &scan_point, VectorScanLines &lines)
{
    int num = scan_point.size();
    int num_lines = num - window_size_ + 1;
    lines.resize( num_lines );
    for(int i = 0; i < num_lines; i++)
    {
        VectorScanPoints subscan(window_size_);
        memcpy( &subscan[0], &scan_point[i], sizeof(ScanPoint)*window_size_);
        fitLinePolar(subscan, lines[i]);
    }
}

bool LineRegressionSegmentation::fitLinePolar(VectorScanPoints &scan, ScanLine &line)
{
    int n = scan.size();
    if(n < window_size_)
    {
        return false;
    }

//        vector<double> costvec, sintvec, cx, cy, devvec, devvec2;
    vector<float> devvec, devvec2;
    float sum_weights = 0, sum_devx = 0, sum_devy = 0, xmw = 0, ymw = 0;
    // Transform ploar to cartesian
    for(int i = 0; i < n; i++)
    {
        devvec.push_back(pow(scan[i].stdrho, -2));
        devvec2.push_back(pow(scan[i].stdrho, 2));
        sum_weights += devvec[i];
        sum_devx += devvec[i] * scan[i].x;
        sum_devy += devvec[i] * scan[i].y;
    }
    xmw = sum_devx / sum_weights;
    ymw = sum_devy / sum_weights;

    // alpha
    float nom = 0, denom = 0, alpha;
    for(int i = 0; i < n; i++)
    {
//            nom += devvec[i]*(cx[i] - xmw)*(cy[i]-ymw);
//            denom += devvec[i]*(pow((cy[i] - ymw), 2) - pow((cx[i]-xmw), 2));
        nom += devvec[i]*(scan[i].x - xmw)*(scan[i].y-ymw);
        denom += devvec[i]*(pow((scan[i].y - ymw), 2) - pow((scan[i].x-xmw), 2));
    }
    nom = -2*nom;
    alpha = 0.5*atan2(nom, denom);

    // r
    float r = xmw*cos(alpha) + ymw*sin(alpha);

    // Eliminate negative radii
    if(r < 0)
    {
        r = -r;
        alpha = alpha + M_PI;
        if(alpha > M_PI)
            alpha -= M_PI*2;
    }
    line.alpha = alpha;
    line.r = r;

    // Computing the WEIGHTED COVARIANCE MATRIX C
    float N = nom, D = denom;
    float NDnom2 = 2*(pow(N,2)+pow(D,2));
    float dr_dalpha = ymw * cos(alpha) - xmw * sin(alpha);

    // sigma_aa vectorized
    float sum_sigma_aa = 0, sigma_aa = 0;// sigma_aa vectorized
    float sum_sigma_rr = 0, sigma_rr = 0;// sigma_rr vectorized
    float sum_sigma_ar = 0, sigma_ar = 0;// sigma_ar vectorized
    for(int i = 0; i < n; i++)
    {
//            ScanPoint &sp = scan[i];
        float dN_drhoi = 2*devvec[i]*(xmw*scan[i].sinPhi + ymw*scan[i].cosPhi - scan[i].rho * sin(2*scan[i].phi));
        float dD_drhoi = 2*devvec[i]*(xmw*scan[i].cosPhi - ymw*scan[i].sinPhi - scan[i].rho * cos(2*scan[i].phi));
        float DdNd = D*dN_drhoi - N*dD_drhoi;
        sum_sigma_aa += pow(DdNd,2)*devvec2[i];

        // sigma_rr
        float dalpha_drhoi = DdNd / NDnom2;
        float dr_drhoi = dalpha_drhoi * dr_dalpha + devvec[i]*cos(scan[i].phi-alpha)/sum_weights;
        sum_sigma_rr += pow(dr_drhoi, 2)*devvec2[i];

        // sigma ar
        sum_sigma_ar += dalpha_drhoi * dr_drhoi * devvec2[i];
    }
    sigma_aa = sum_sigma_aa / pow(NDnom2,2);
    sigma_rr = sum_sigma_rr;
    sigma_ar = sum_sigma_ar;

    line.saa = sigma_aa;
    line.srr = sigma_rr;
    line.sar = sigma_ar;

    return true;
}

float LineRegressionSegmentation::calcCompactness(VectorScanLines &lines)
{
    int n = lines.size();

    // Step 1: transform alphas to eliminate cyclic alpha-axis
    // The values are shifted such that their minimum lies on zero.
    float minalpha = lines[0].alpha;
    for(int i = 1; i < n; i++)
    {
        if(minalpha > lines[i].alpha)
            minalpha = lines[i].alpha;
    }

    for(int i = 0; i < n; i++)
    {
        float diffi = lines[i].alpha - minalpha;
        if(diffi > M_PI)
            lines[i].alpha = diffi - 2*M_PI;
        else
            lines[i].alpha = diffi;
    }

    // Step 2: stack and apply multivariate mean
    vector<Eigen::Vector2f> xv(n);
    vector<Eigen::Matrix2f> Cv(n);
    for(int i = 0; i < n; i++)
    {
        ScanLine &line = lines[i];
        xv[i] << line.alpha, line.r;
        Cv[i] << line.saa, line.sar, line.sar, line.srr;
    }
    Eigen::Vector2f xw;
    Eigen::Matrix2f Cw;
    meanwm(xv, Cv, xw, Cw);

    // Step 3: sum all mahalanobis distances
    float accu = 0;
    ScanLine l2(xw(0), xw(1));
    for(int i = 0; i < n; i++)
    {
        accu += mahalanobisar(lines[i], l2);
    }

    return accu;
}

void LineRegressionSegmentation::findLineRegions(vector<float> &compactness, float threshfidel, VectorScanSegments &indices)
{
    int begin = 0;
    int end = 0;
    int min_inlier = min_inlier_;

    for(int i = 0; i < compactness.size(); i++)
    {
        if(compactness[i] < threshfidel)
        {
            end = i;
        }
        else
        {
//            cout <<BLUE << " " << distances[i];
            int length = end - begin + 1;
            if( length >= min_inlier)
            {
                ScanSegment seg;
                seg.begin = begin;
                seg.end = end;
                seg.size = end - begin + 1;
                indices.push_back( seg );
            }
            begin = i;
        }
    }
    int length = end - begin + 1;
    if( length >= min_inlier)
    {
        ScanSegment seg;
        seg.begin = begin;
        seg.end = end;
        seg.size = end - begin + 1;
        indices.push_back( seg );
    }
}

void LineRegressionSegmentation::refineRegion(VectorScanSegments &indices)
{
    int bias = (window_size_ - 1) / 2 * 2 + 2;
    for(int i = 0; i < indices.size(); i++)
    {
        indices[i].end += bias;
        indices[i].size = indices[i].end - indices[i].begin + 1;
    }
}

void LineRegressionSegmentation::meanwm(vector<Eigen::Vector2f> xv, vector<Eigen::Matrix2f> Cv,
                            Eigen::Vector2f &xw, Eigen::Matrix2f &Cw)
{
    int n = xv.size();
    Eigen::Matrix2f Caccu;
    Eigen::Vector2f xaccu;
    Caccu.setZero();
    xaccu.setZero();

    for(int i = 0; i < n; i++)
    {

        Eigen::Matrix2f invCi = Cv[i].inverse();
        Caccu += invCi;
        xaccu += invCi*xv[i];
    }

    Cw = Caccu.inverse();
    xw = Cw*xaccu;
}

float LineRegressionSegmentation::mahalanobisar(ScanLine l1, ScanLine l2)
{
    // Step 1: Take diffenrece in angle

    // Normalize angle a1,a2
    double a1 = l1.alpha, a2 = l2.alpha;

    if(a1 >= 2*M_PI)
    {
        a1 -= 2*M_PI;
        while(a1 >= 2*M_PI)
        {
            a1 -= 2*M_PI;
        }
    }
    else if(a1 < 0)
    {
        a1 += 2*M_PI;
        while(a1 < 0)
        {
            a1+= 2*M_PI;
        }
    }

    if(a2 >= 2*M_PI)
    {
        a2 -= 2*M_PI;
        while(a2 >= 2*M_PI)
        {
            a2 -= 2*M_PI;
        }
    }
    else if(a2 < 0)
    {
        a2 += 2*M_PI;
        while(a2 < 0)
        {
            a2+= 2*M_PI;
        }
    }

    // Take difference and unwrap
    double dalpha = a1 - a2;
    if(a1 > a2)
    {
        while(dalpha > M_PI)
        {
            dalpha -= 2*M_PI;
        }
    }
    else if(a1 < a2)
    {
        while(dalpha < (-M_PI))
        {
            dalpha += 2*M_PI;
        }
    }

    // Step 2: Take difference in r
    double dr = l1.r - l2.r;

    // Step 3: Put together innovation and innovation covariance
    Eigen::Vector2d v(dalpha, dr);
    double dinv = (l1.saa + l2.saa)*(l1.srr + l2.srr) - (l1.sar + l2.sar)*(l1.sar+ l2.sar);
    Eigen::Matrix2d Sinv;
    Sinv << (l1.srr+l2.srr), (-l1.sar-l2.sar), (-l1.sar-l2.sar), (l1.saa+l2.saa);
    Sinv = Sinv/dinv;

    // Step 4: Calculate (general) Mahalanobis distance
    double d = v.transpose()*Sinv*v;

    return d;
}

}
