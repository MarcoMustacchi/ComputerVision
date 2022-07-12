	
	
	
void detectOverlap(std::vector<std::vector<int>> coordinates_bb;)
{	
	
	//__________________________ Detect if overlap _________________________________// 
	    
	bool overlap = 0;
	
	int x1, y1, width1, height1;
	int x2, y2, width2, height2;
	
	for (int i = 0; i < n_hands-1; i++) 
	{
	    x1 = coordinates_bb[i][0];
	    y1 = coordinates_bb[i+1][1];
	    width1 = coordinates_bb[i+2];
	    height1 = coordinates_bb[3+i];
	    x2 = coordinates_bb[i+1][0];  // attenzione, potrebbe essere sbagliato
	    y2 = coordinates_bb[i+1][1];
	    width2 = coordinates_bb[i+1][2];
	    height2 = coordinates_bb[i+1][3];
	    
	    overlap = detectOverlapSegmentation(x1, y1, width1, height1, x2, y2, width2, height2); 
	    
	}
	
	// se controllo ogni combinazione, tengo il conto ed e' uguale a numero di mani -> no Overlap, inserisco in nuovo vettore
	// se trovo overlap, inserisco in nuovo vettore come somma tra i due, tenendo bounding box grande
	// il tutto in un while finche' bool overlap e' Falso (non c'e' neanche un Overlap)
	
	std::cout << "Overlap " << overlap << std::endl;
	
}


bool detectOverlapSegmentation(int x, int y, int width, int height, int a, int b, int c, int d)
{

    // intersection region
    int xA = std::max(x, a);
    int yA = std::max(y, b);
    int xB = std::min(x+width, a+c);
    int yB = std::min(y+height, b+d);
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    std::cout << "Intersection area is " << interArea << std::endl;
    
    bool overlap = 0;
    
    if (interArea != 0)
        overlap = 1;
    
    return overlap;
    
}


    // Union of the overlap
    int xA = std::min(x, a);
    int yA = std::min(y, b);
    int xB = std::max(x+width, a+c);
    int yB = std::max(y+height, b+d);



