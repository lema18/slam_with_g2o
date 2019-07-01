#include <stdio.h>
#include <iostream>
#include <ctype.h>
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
#include <algorithm>
#include <iterator> 
#include <vector>
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
    if(argc< 12)
    {
        cout<< "bad input\n";
        cout << "please enter:\n";
        cout << "argv[1]= path to rgb images\n";
        cout << "argv[2]= number of images to compose the initial map\n";
        cout << "argv[3]= threshold for points tha are seen in more than i images\n";
        cout << "argv[4]= distance ratio to reject features\n";
        cout << "argv[5]= confidence on ransac\n";
        cout << "argv[6]= threshold for error_reprojection on ransac\n";
        cout << "argv[7]= g2o iterations\n";
        cout << "argv[8]= use dense solver(1 to set or 0 to disable)\n";
        cout << "argv[9]= use robust Kernel(1 to set or 0 to disable)\n";
        cout << "argv[10]= Nº of known poses\n";
        cout << "argv[11]= Nº of vertices to get fixed in g2o solver\n";
        exit(-1);
    }
    //init calibration matrices
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinseca = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinseca.convertTo(intrinseca, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    double focal_length=516.9;
    Eigen::Vector2d principal_point(318.6,255.3); 
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
    //g2o iterations
    int niter;
    sscanf(argv[7],"%d",&niter);
    //use dense solver
    int dense;
    sscanf(argv[8],"%d",&dense);
    //use robust Kernel
    int robust_kernel;
    sscanf(argv[9],"%d",&robust_kernel);
    //known poses
    int known_poses;
    sscanf(argv[10],"%d",&known_poses);
    int cam_fixed;
    sscanf(argv[11],"%d",&cam_fixed);
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;
    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;
    //map for match index
    unordered_map<int,vector<int>> match_index;
    //map for 3d_point initialization
    unordered_map<int,Eigen::Vector3d> init_3d;
    //iterator to iterate through images
    int l=0;
    //identifier for 3d_point
    int ident=0;
    //we need variables to store the last image and the last features & descriptors
    auto pt =SURF::create();
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
                double z=0.5; //initial z invented
                Eigen::Vector3d init_guess;
                init_guess[0]=((search_point->second[0].x - principal_point[0])/focal_length)*z;
                init_guess[1]=((search_point->second[0].y - principal_point[1])/focal_length)*z;
                init_guess[2]=z;
                init_3d[i]=init_guess;
            }
            else
            {
                valid_points[i]=0;
            }
        }
    }
    //3d points to optimize
    vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > camera_poses;
    g2o::CameraParameters * cam_params = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);
    //timestamp tx ty tz qx qy qz qw
    //initialization of 50 vertices, 2 of them fixed
    int vertex_id=0;
    double aux;
    string line;
    ifstream myfile ("../first_50.txt");
    Eigen::Matrix4d initPose = Eigen::Matrix4d::Identity();
    if (myfile.is_open())
    {
        for(int i=0;i<nImages;i++)
        {
            g2o::VertexSE3Expmap * v_se3=new g2o::VertexSE3Expmap();
            Eigen::Quaterniond qa;
            g2o::SE3Quat pose;
            Eigen::Vector3d trans;
            Eigen::Matrix4d poseMatrix;
            if(i<known_poses)
            {
                for(int j=0;j<8;j++)
                {
                    if(j==1) trans[0]=0;
                    if(j==2) trans[1]= 0;
                    if(j==3) trans[2]= 0;
                    if(j==4) qa.x()= 0;
                    if(j==5) qa.y()= 0;
                    if(j==6) qa.z()= 0;
                    if(j==7) qa.w()= 1;
                }
       
                std::cout << poseMatrix << std::endl;
                qa = Eigen::Quaterniond(poseMatrix.block<3,3>(0,0));
                trans = poseMatrix.block<3,1>(0,3);

                pose=g2o::SE3Quat(qa,trans);
                v_se3->setId(vertex_id);
                v_se3->setEstimate(pose);
                if(i<cam_fixed)
                {
                    v_se3->setFixed(true);
                }
                optimizer.addVertex(v_se3);
                camera_poses.push_back(pose);
            }
            else
            {
                pose=camera_poses[known_poses-1];
                v_se3->setId(vertex_id);
                v_se3->setEstimate(pose);
                optimizer.addVertex(v_se3);
                camera_poses.push_back(pose);
            }
            vertex_id++;
        }
        myfile.close();
    }   
    int point_id=vertex_id;
    for(int j=0;j<ident;j++)
    {
        if(valid_points[j]==1)
        {
            g2o::VertexSBAPointXYZ * v_p= new g2o::VertexSBAPointXYZ();
            v_p->setId(point_id);
            v_p->setMarginalized(true);
            v_p->setEstimate(init_3d[j]);
            optimizer.addVertex(v_p);
            vector<Point2f> aux_pt;
            vector<int> aux_im;
            aux_pt=pt_2d[j];
            aux_im=img_index[j];
            //we search point j on image i
            for(int p=0;p<aux_im.size();p++)
            {
                //we add the edge connecting the vertex of camera position and the vertex point
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(aux_im[p])->second));
                
                Eigen::Vector2d measurement(aux_pt[p].x,aux_pt[p].y);
                e->setMeasurement(measurement);
                
                e->information() = Eigen::Matrix2d::Identity();
                
                if (robust_kernel)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                }
                e->fx = 517.3; 
                e->fy =  516.5; 
                e->cx =  318.6; 
                e->cy = 255.3;

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
    optimizer.save("test.g2o");
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
        g2o::SE3Quat updated_pose;
        Eigen::Matrix4f eig_cam_pos = Eigen::Matrix4f::Identity();
        Eigen::Quaterniond cam_quat;
        Eigen::Matrix3d cam_rotation;
        Eigen::Vector3d cam_translation;
        g2o::HyperGraph::VertexIDMap::iterator pose_it= optimizer.vertices().find(i);
        g2o::VertexSE3Expmap * v_se3= dynamic_cast< g2o::VertexSE3Expmap * >(pose_it->second);
        updated_pose=v_se3->estimate();
        cam_translation=updated_pose.translation();
        cam_quat=updated_pose.rotation();
        cam_rotation=cam_quat.normalized().toRotationMatrix();
        eig_cam_pos.block<3,3>(0,0) = cam_quat.matrix().cast<float>();
        eig_cam_pos.block<3,1>(0,3) = cam_translation.cast<float>();
        
        // eig_cam_pos(0, 0) = cam_rotation(0,0);
		// eig_cam_pos(0, 1) = cam_rotation(0,1);
		// eig_cam_pos(0, 2) = cam_rotation(0,2);
		// eig_cam_pos(0, 3) = cam_translation[0];
		// eig_cam_pos(1, 0) = cam_rotation(1,0);
		// eig_cam_pos(1, 1) = cam_rotation(1,1);
		// eig_cam_pos(1, 2) = cam_rotation(1,2);
		// eig_cam_pos(1, 3) = cam_translation[1];
		// eig_cam_pos(2, 0) = cam_rotation(2,0);
		// eig_cam_pos(2, 1) = cam_rotation(2,1);
		// eig_cam_pos(2, 2) = cam_rotation(2,2);
		// eig_cam_pos(2, 3) = cam_translation[2];
		// eig_cam_pos(3, 0) = 0;
		// eig_cam_pos(3, 1) = 0;
		// eig_cam_pos(3, 2) = 0;
		// eig_cam_pos(3, 3) = 1;

        cam_pos = eig_cam_pos.inverse();
        viewer.addCoordinateSystem(0.05, cam_pos, name);
		pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		viewer.addText3D(std::to_string(i), textPoint, 0.01, 1, 1, 1, "text_"+std::to_string(i));

        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

        file << i << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    
    }
    
    file.close();
	pcl::PointCloud<pcl::PointXYZ> cloud;
    for(int j=0;j<cnt;j++)
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
