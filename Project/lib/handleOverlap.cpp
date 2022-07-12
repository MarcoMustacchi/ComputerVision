
	/*
	//__________________________ Detect if overlap _________________________________// 
	    
	bool overlap = 0;
	
	int x1, y1, width1, height1;
	int x2, y2, width2, height2;
	
	int temp2 = 3;
	
	// attenzione, questo ciclo mi fa solo un controllo se aggiungo i+4
	for (int i = 0; i < n_hands; i+=4) // ciclo for per controllo tutte le combinazioni di bounding box
	{
	    x1 = coordinates_bb[0+i];
	    y1 = coordinates_bb[1+i];
	    width1 = coordinates_bb[2+i];
	    height1 = coordinates_bb[3+i];
	    x2 = coordinates_bb[1+i+temp2];  // attenzione, potrebbe essere sbagliato
	    y2 = coordinates_bb[2+i+temp2];
	    width2 = coordinates_bb[3+i+temp2];
	    height2 = coordinates_bb[4+i+temp2];
	    
	    overlap = detectOverlapSegmentation(x1, y1, width1, height1, x2, y2, width2, height2);  
	    
	    temp2 = temp2 + 3;
	}
	
	std::cout << "Overlap " << overlap << std::endl;
	
	*/
	
	/*
	bool overlap = 1;
	
	//______________________________ Handle Overlap between masks __________________________//
		
	cv::Mat mask_Overlap1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat mask_Overlap2(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	
	int i = 0;
	mask_final_ROI[i].copyTo(mask_Overlap1(cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
	
	i = 1;
	mask_final_ROI[i].copyTo(mask_Overlap2(cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
		

	cv::namedWindow("mask_final_ROI 1");
	cv::imshow("mask_final_ROI 1", mask_final_ROI[0]);
	cv::namedWindow("mask_final_ROI 2");
	cv::imshow("mask_final_ROI 2", mask_final_ROI[1]);
	cv::waitKey(0);
	
	cv::Mat mask_Intersection(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::bitwise_and(mask_Overlap1, mask_Overlap2, mask_Intersection);
	cv::namedWindow("mask_Intersection");
	cv::imshow("mask_Intersection", mask_Intersection);
	cv::waitKey(0);
	
	
	cv::destroyAllWindows();

	
	if (overlap == 1) 
	{
	    int smaller = 1;
    	if (smaller == 1) // piu piccola la prima ROI
		    mask_Overlap1 = mask_Overlap1 - mask_Intersection;
	    else 
		    mask_Overlap2 = mask_Overlap2 - mask_Intersection;
		    
		cv::namedWindow("mask_Overlap1");
	    cv::imshow("mask_Overlap1", mask_Overlap1);
	    cv::namedWindow("mask_Overlap2");
	    cv::imshow("mask_Overlap2", mask_Overlap2);
	    cv::waitKey(0);
	}
		
	cv::Mat mask_final_Overlap(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::bitwise_or(mask_Overlap1, mask_Overlap2, mask_final_Overlap);
	
	cv::namedWindow("mask_final_Overlap");
	cv::imshow("mask_final_Overlap", mask_final_Overlap);
	cv::waitKey(0);
	
	cv::imwrite("../results/mask_Overlap.png", mask_final_Overlap);
	
	cv::Mat mask_Opening;
	
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10));
	
	cv::morphologyEx(mask_final_Overlap, mask_Opening, cv::MORPH_OPEN, kernel);
		
	cv::namedWindow("mask_final_Opening");
	cv::imshow("mask_final_Opening", mask_final_Overlap);
	cv::waitKey(0);
	*/
	
