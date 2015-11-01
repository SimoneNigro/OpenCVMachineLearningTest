// Example : random forest (tree) learning
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : opticaldigits_ex

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <opencv/cv.h>       // opencv general include file
#include <opencv/ml.h>		  // opencv machine learning include file
//in order to compile i had to add opencv/ in front of the two header files

//to compile, add -lopencv_core and -lopencv_ml to the g++ command

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
/******************************************************************************/
// global definitions (for speed and ease of use)
/*
#define NUMBER_OF_TRAINING_SAMPLES 500
//Attributes are:  
//Recency - months since last donation), Frequency - total number of donation), Monetary - total blood donated in c.c.), Time - (months since first donation)
#define ATTRIBUTES_PER_SAMPLE 4

#define NUMBER_OF_TESTING_SAMPLES 248

//Classes: hasn't donated in march 2007 (0) / has donated (1) 
#define NUMBER_OF_CLASSES 2

// N.B. classes are integer handwritten digits in range 0-9
*/
/******************************************************************************/
//tokenizer
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}
//returns the number of lines given a file
int number_of_lines(char* filename)
{
	FILE* myfile = fopen(filename, "r");
	int ch, number_of_lines = 0;

	do 
	{
	    ch = fgetc(myfile);
	    if(ch == '\n')
	    	number_of_lines++;
	} while (ch != EOF);

	// last line doesn't end with a new line!
	// but there has to be a line at least before the last line
	if(ch != '\n' && number_of_lines != 0) 
	    number_of_lines++;

	fclose(myfile);
	
	return number_of_lines;
}



//returns the number of training samples, the number of testing samples and the attributes per sample given the 2 files
int* find_parameters_from_csv(char* filename_train, char* filename_test)
{
    int number_of_training_samples = 0;
    int number_of_testing_samples = 0;
    int number_of_attributes;
    int* results;  
    vector<string> attributes;

    FILE* f1 = fopen( filename_train, "r" );
    FILE* f2 = fopen( filename_test, "r" );
    
    if( !f1 )
    {
        printf("ERROR: cannot read file %s\n",  filename_train);
        return 0; // all not OK
    }
    if( !f2 )
    {
        printf("ERROR: cannot read file %s\n",  filename_test);
        return 0; // all not OK
    }


    number_of_training_samples = number_of_lines(filename_train);
    number_of_testing_samples = number_of_lines(filename_test);

    std::ifstream input(filename_train);
    string line;

    getline( input, line );
    //fa il trim
    line.erase(line.find_last_not_of(" \n\r\t")+1);

    attributes = split(line, ',');
 
    number_of_attributes = attributes.size() - 1;

    results[0] = number_of_training_samples;
    results[1] = number_of_testing_samples;
    results[2] = number_of_attributes;

    return results;
}

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(char* filename, Mat data, Mat classes,
                       int n_samples, int ATTRIBUTES_PER_SAMPLE)
{
    float tmp;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {

        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < ATTRIBUTES_PER_SAMPLE)
            {

                // first 4 elements (0-3) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
                // printf("%f,", data.at<float>(line, attribute));

            }
            else if (attribute == ATTRIBUTES_PER_SAMPLE)
            {

                // attribute 5 is the class label {0 ... 1}

                fscanf(f, "%f,", &tmp);
                classes.at<float>(line, 0) = tmp;
                // printf("%f\n", classes.at<float>(line, 0));

            }
        }
    }

    fclose(f);

    return 1; // all OK
}

/******************************************************************************/

int main( int argc, char** argv )
{
    // lets just check the version first

    printf ("OpenCV version %s (%d.%d.%d)\n",
            CV_VERSION,
            CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    
    if(argc != 4)
    {
     	printf("Usage: %s file_training file_testing number_of_classes", argv[0]);
        return 0;
    }

    //define number of training and testing samples and number of attributes
    int* results = find_parameters_from_csv(argv[1], argv[2]);
    
    int NUMBER_OF_TRAINING_SAMPLES = results[0];
    int NUMBER_OF_TESTING_SAMPLES = results[1];
    int ATTRIBUTES_PER_SAMPLE = results[2];

    int NUMBER_OF_CLASSES = atoi(argv[3]);

    printf("N° of training samples: %d \nN° testing of samples: %d \nN° of attributes: %d \nN° of classes: %d \n", NUMBER_OF_TRAINING_SAMPLES,NUMBER_OF_TESTING_SAMPLES,ATTRIBUTES_PER_SAMPLE,NUMBER_OF_CLASSES );

    // define training data storage matrices (one for attribute examples, one
    // for classifications)

    Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //define testing data storage matrices

    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    double result; // value returned from a prediction

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE))
    {
        // define the parameters for training the random forest (trees)

        float priors[] = {1,1,1,1};  // weights of each classification for classes
        // (all equal as equal samples of each digit)

        CvRTParams params = CvRTParams(25, // max depth
                                       2, // min sample count
                                       0, // regression accuracy: N/A here
                                       false, // compute surrogate split, no missing data
                                       15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                       priors, // the array of priors
                                       false,  // calculate variable importance
                                       4,       // number of variables randomly selected at node and used to find the best split(s).
                                       100,	 // max number of trees in the forest
                                       0.01f,				// forrest accuracy
                                       CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                      );

        // train random forest classifier (using training data)

        printf( "\nUsing training database: %s\n\n", argv[1]);
        CvRTrees* rtree = new CvRTrees;

        rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                     Mat(), Mat(), var_type, Mat(), params);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES];

	//initialize every element in false_positives to 0
	for (int z = 0; z < NUMBER_OF_CLASSES; z++)
        {
		false_positives[z] = 0;
	}

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // run random forest prediction

            result = rtree->predict(test_sample, Mat());

            printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)

            if (fabs(result - testing_classifications.at<float>(tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                false_positives[(int) result]++;

            }
            else
            {

                // otherwise correct

                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classifications: %d (%g%%)\n",
                argv[2],
                correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
                wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

        for (int i = 0; i < NUMBER_OF_CLASSES; i++)
        {
            printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }


        // all matrix memory free by destructors


        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/

