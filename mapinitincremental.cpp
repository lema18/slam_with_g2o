#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include<stdlib.h>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <fstream>
#include <Eigen/StdVector>

#include <unordered_set>
#include <stdint.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
//#include "g2o/math_groups/se3quat.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs) {
	stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss << _folder << "/left_" << _number << ".png";
	std::cout << "Loading image: " << ss.str() << std::endl;
	Mat image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
	cvtColor(image, image, COLOR_BGR2GRAY);
	cv::Mat image_u;
	undistort(image, image_u, _intrinsics, _coeffs);
	return image_u;
}

bool matchFeatures(	vector<KeyPoint> &_features1, cv::Mat &_desc1, 
					vector<KeyPoint> &_features2, cv::Mat &_desc2,
					vector<int> &_ifKeypoints, vector<int> &_jfKeypoints,double dst_ratio,double confidence,double reproject_err){

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	vector<Point2d> source, destination;
	vector<uchar> mask;
	vector<int> i_keypoint, j_keypoint;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{
			source.push_back(_features1[matches[k][0].queryIdx].pt);
			destination.push_back(_features2[matches[k][0].trainIdx].pt);
			i_keypoint.push_back(matches[k][0].queryIdx);
			j_keypoint.push_back(matches[k][0].trainIdx);
		}
	}

	//aplicamos filtro ransac
	findFundamentalMat(source, destination, FM_RANSAC,reproject_err, confidence, mask);
	for (int m = 0; m < mask.size(); m++)
	{
		if (mask[m])
		{
			_ifKeypoints.push_back(i_keypoint[m]);
			_jfKeypoints.push_back(j_keypoint[m]);
		}
	}
}

void displayMatches(	cv::Mat &_img1, std::vector<cv::KeyPoint> &_features1, std::vector<int> &_filtered1,
						cv::Mat &_img2, std::vector<cv::KeyPoint> &_features2, std::vector<int> &_filtered2){
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);

	for(unsigned i = 0; i < _filtered1.size(); i++){
		auto p1 = _features1[_filtered1[i]].pt;
		auto p2 = _features2[_filtered2[i]].pt + cv::Point2f(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}

	cv::imshow("display", display);
	cv::waitKey(3);
}
int main(int argc,char ** argv)
{
    //init calibration matrices
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinseca = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinseca.convertTo(intrinseca, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    //number of images to compose the initial map
    int nImages =  atoi(argv[2]);
    //threshold for points tha are seen in more than i images
    int img_threshold;
    sscanf(argv[3],"%d",&img_threshold);
    //distance ratio to reject features
    double dst_ratio;
    sscanf(argv[4],"%lf",&dst_ratio);
    //confidence on ransac
    double confidence;
    sscanf(argv[5],"%lf",&confidence);
    //threshold for error_reprojection on ransac
    double reproject_err;
    sscanf(argv[6],"%lf",&reproject_err);
    //cvsba iterations
    int niter;
    sscanf(argv[7],"%d",&niter);
    //use dense solver
    int dense;
    sscanf(argv[8],"%d",&dense);
    //use robust Kernel
    int robust_kernel;
    sscanf(argv[9],"%d",&robust_kernel);
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;
    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;
    //map for match index
    unordered_map<int,vector<int>> match_index;
    //iterator to iterate through images
    int l=0;
    //identifier for 3d_point
    int ident=0;
    //we need variables to store the last image and the last features & descriptors
    auto pt =SURF::create(500);
    Mat foto1_u;
    vector<KeyPoint> features1;
    Mat descriptors1;
    //load first image
    foto1_u = loadImage(argv[1], l, intrinseca, distcoef);
    pt->detectAndCompute(foto1_u, Mat(), features1, descriptors1);
    l++;
    while(l<nImages)
    {   
        //load new image
        Mat foto2_u = loadImage(argv[1], l, intrinseca, distcoef);
        //create pair of features
        vector<KeyPoint> features2;
	    Mat descriptors2;
	    pt->detectAndCompute(foto2_u, Mat(), features2, descriptors2);
        //match features
        vector<int> if_keypoint, jf_keypoint;
        matchFeatures(features1, descriptors1, features2, descriptors2, if_keypoint, jf_keypoint,dst_ratio,confidence,reproject_err);
        displayMatches(foto1_u, features1, if_keypoint,foto2_u, features2, jf_keypoint);
        Mat used_features=Mat::zeros(1,int(if_keypoint.size()),CV_64F);//to differentiate the features that correspond to new points from those that do not
        if(ident>0)
        {
            for(int j=0;j<ident;j++)
            {
                auto search_match=match_index.find(j);
                auto search_img=img_index.find(j);
                if(search_match!=match_index.end() && search_img!=img_index.end())
                {
                    auto it_match=search_match->second.end();
                    it_match--;
                    auto it_img=search_img->second.end();
                    it_img--;
                    int last_match=*it_match;
                    int last_img=*it_img;
                    int flag=0;
                    for(int k=0;k<if_keypoint.size() && !flag;k++)
                    {
                        if(if_keypoint[k]==last_match && last_img==l-1)
                        {
                            //we add the new projection for the same 3d point
                            pt_2d[j].push_back(features2[jf_keypoint[k]].pt);
                            img_index[j].push_back(l);
                            match_index[j].push_back(jf_keypoint[k]);
                            used_features.at<double>(k)=1;
                            flag=1;
                        }
                    }
                }
            }
        }
        //we add the projections of the new 3d points for two consecutive frames
       
        for (int i=0;i<if_keypoint.size();i++)
        {
            if(used_features.at<double>(i)==0)
            {
                pt_2d[ident]=vector<Point2f>();
                pt_2d[ident].push_back(features1[if_keypoint[i]].pt);
                img_index[ident]=vector<int>();
                img_index[ident].push_back(l-1);
                match_index[ident]=vector<int>();
                match_index[ident].push_back(if_keypoint[i]);
                pt_2d[ident].push_back(features2[jf_keypoint[i]].pt);
                img_index[ident].push_back(l);
                match_index[ident].push_back(jf_keypoint[i]);
                ident++;
            }
        }
        foto1_u=foto2_u;
        features1=features2;
        descriptors1=descriptors2;
        used_features.release();
        l++;
	}
    //prepare varibles for g2o optimization
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (dense)
    {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }
    else
    {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);
    g2o::ParameterSE3Offset* cameraOffset = new g2o::ParameterSE3Offset;
    cameraOffset->setId(0);
    optimizer.addParameter(cameraOffset);
    //filter to reject points that are not visible in more than 3 images
    int valid_points[ident];
    //debug cnt
    int cnt=0;
    for(int i=0;i<ident;i++)
    {
        auto search_point=pt_2d.find(i);
        if(search_point!=pt_2d.end())
        {
            int dimension= search_point->second.size();
            if(dimension>=img_threshold)
            {
                valid_points[i]=1;
                cnt++;
            }
            else
            {
                valid_points[i]=0;
            }
        }
    }
    //3d points to optimize
    vector<Eigen::Vector3d> points_3d;
    for (int i=0;i<ident;i++)
    {
        points_3d.push_back(Eigen::Vector3d(0,0,0.5));
    }
    Eigen::Vector2d focal_length(517.3,516.5);
    Eigen::Vector2d principal_point(318.6,255.3); 
    vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d> > camera_poses;
    g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],principal_point[0],principal_point[1],0.); 
    //initialization of 50 vertices, 2 of them fixed
    int vertex_id=0;
    for(int i=0;i<nImages;i++)
    {
        g2o::VertexSCam * v_se3=new g2o::VertexSCam();
        Eigen::Quaterniond qa;
        Eigen::Isometry3d pose;
        Eigen::Vector3d trans;
        if(i==0)
        {
            //first vertex from ground_truth
            qa.x()=0.6574;
            qa.y()=0.6126;
            qa.z()=-0.2949;
            qa.w()=-0.3248;
            trans[0]=1.3405;
            trans[1]=0.6266;
            trans[2]=1.6575;
            pose = qa;
            pose.translation() = trans;
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            v_se3->setAll();
            v_se3->setFixed(true);
            optimizer.addVertex(v_se3);
            camera_poses.push_back(pose);
            vertex_id++;
            continue;
        }
        //rest of vertices are initialized on the second camera pose from ground_truth
        qa.x()=0.6579;
        qa.y()=0.6161;
        qa.z()=-0.2932;
        qa.w()=-0.3189;
        trans[0]=1.3303;
        trans[1]=0.6256;
        trans[2]=1.6464;
        pose = qa;
        pose.translation() = trans;
        v_se3->setId(vertex_id);
        v_se3->setEstimate(pose);
        v_se3->setAll();
        if(i==1)
        {
            v_se3->setFixed(true);
        }
        optimizer.addVertex(v_se3);
        camera_poses.push_back(pose);
        vertex_id++;
    }

    int point_id=vertex_id;
    for(int j=0;j<ident;j++)
    {
        if(valid_points[j]==1)
        {
            g2o::VertexSBAPointXYZ * v_p= new g2o::VertexSBAPointXYZ();
            v_p->setId(point_id);
            v_p->setMarginalized(true);
            v_p->setEstimate(points_3d.at(j));
            optimizer.addVertex(v_p);
            vector<Point2f> aux_pt;
            vector<int> aux_im;
            aux_pt=pt_2d[j];
            aux_im=img_index[j];
            int stop_flag=0;
            //we search point j on image i
            for(int p=0;p<aux_im.size();p++)
            {
                //we add the edge connecting the vertex of camera position and the vertex point
                Eigen::Vector2d z(aux_pt[p].x,aux_pt[p].y);
                g2o::EdgeProjectXYZ2UV * e= new g2o::EdgeProjectXYZ2UV();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(aux_im[p])->second));
                e->setMeasurement(z);
                e->information() = Eigen::Matrix2d::Identity();
                if (robust_kernel)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                }
                e->setParameterId(0, 0);
                optimizer.addEdge(e);
            }
            point_id++;
        }
    }
    cout << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(niter);

	/* Graphical representation of camera's position and 3d points*/
	pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35);

    mkdir("./lectura_datos", 0777);
    std::ofstream file("./lectura_datos/odometry.txt");
    if (!file.is_open()) return -1;

	for (int i = 0; i < nImages; i++)
    {
		stringstream sss;
		string name;
		sss << i;
		name = sss.str();
		Eigen::Affine3f cam_pos;
        Eigen::Isometry3d updated_pose;
        Eigen::Matrix4f eig_cam_pos;
        Eigen::Matrix3d cam_rotation;
        Eigen::Vector3d cam_translation;
        g2o::HyperGraph::VertexIDMap::iterator pose_it= optimizer.vertices().find(i);
        g2o::VertexSCam * v_se3= dynamic_cast< g2o::VertexSCam * >(pose_it->second);
        updated_pose=v_se3->estimate();
        cam_rotation=updated_pose.linear();
        cam_translation=updated_pose.translation();
        eig_cam_pos(0, 0) = cam_rotation(0,0);
		eig_cam_pos(0, 1) = cam_rotation(0,1);
		eig_cam_pos(0, 2) = cam_rotation(0,2);
		eig_cam_pos(0, 3) = cam_translation[0];
		eig_cam_pos(1, 0) = cam_rotation(1,0);
		eig_cam_pos(1, 1) = cam_rotation(1,1);
		eig_cam_pos(1, 2) = cam_rotation(1,2);
		eig_cam_pos(1, 3) = cam_translation[1];
		eig_cam_pos(2, 0) = cam_rotation(2,0);
		eig_cam_pos(2, 1) = cam_rotation(2,1);
		eig_cam_pos(2, 2) = cam_rotation(2,2);
		eig_cam_pos(2, 3) = cam_translation[2];
		eig_cam_pos(3, 0) = 0;
		eig_cam_pos(3, 1) = 0;
		eig_cam_pos(3, 2) = 0;
		eig_cam_pos(3, 3) = 1;

        cam_pos=eig_cam_pos;
        viewer.addCoordinateSystem(0.05, cam_pos, name);
		pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		viewer.addText3D(std::to_string(i), textPoint, 0.01, 1, 1, 1, "text_"+std::to_string(i));

        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

        file << 0.0 << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    
    }
    
    file.close();

	pcl::PointCloud<pcl::PointXYZ> cloud;
    for(int j=0;j<points_3d.size();j++)
    {
        g2o::HyperGraph::VertexIDMap::iterator point_it= optimizer.vertices().find(vertex_id+j);
        g2o::VertexSBAPointXYZ * v_p= dynamic_cast< g2o::VertexSBAPointXYZ * > (point_it->second);
        Eigen::Vector3d p_aux=v_p->estimate();
        pcl::PointXYZ p(p_aux[0], p_aux[1], p_aux[2]);
        cloud.push_back(p);
    }

	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");

	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}