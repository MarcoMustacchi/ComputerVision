
/**
 * @file write_to_file.h
 *
 * @brief  write_to_file Header file
 *
 * @author Marco Mustacchi
 *
 */

#ifndef WRITE2FILE_H
#define WRITE2FILE_H


void write_results_Detection(const std::vector<std::vector<int>>& new_coordinates_bb, std::string image_number);
void write_performance_Detection(float value, std::string image_number);
void write_performance_Segmentation(float value, std::string image_number);


#endif
