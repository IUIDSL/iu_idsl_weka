package edu.indiana.sice.idsl.weka;

import java.io.*;

import weka.core.*; // Instances, Attribute
import weka.core.converters.*; // ArffLoader, CSVLoader, ArffSaver;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

/**
	@author Jeremy Yang, Abhik Seal
*/
public class forest_test
{
  public static String RandomForest_test(File ftrain_csv,File feval_csv)
	throws Exception
  {
    Instances dataset_train=weka_utils.LoadCSV(ftrain_csv);
    Instances dataset_eval=weka_utils.LoadCSV(feval_csv);
    //Instances dataset_train=LoadARFF(feval_arff);
    //Instances dataset_eval=LoadARFF(feval_arff);

    RandomForest rdfr = weka_utils.RandomForest_build(dataset_train);
   
    // Prints the model file in the source directory       
    //ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("data/rdfr.model"));
    //oos.writeObject(rdfr);
    //oos.flush();
    //oos.close();

    Evaluation eval=weka_utils.RandomForest_Evaluation(rdfr, dataset_train, dataset_eval);

    String outstr = weka_utils.RandomForest_Eval2Txt(eval);
    return outstr;
  }

  public static void main(String[] args) throws Exception
  {
    String outstr = RandomForest_test(
	new File("/home/jjyang/weka/abhik/data/6510train.csv"),
	new File("/home/jjyang/weka/abhik/data/6510test.csv"));

    System.out.println(outstr);
  }
}
