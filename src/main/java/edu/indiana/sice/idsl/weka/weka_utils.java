package edu.indiana.sice.idsl.weka;

import java.io.*;

import weka.core.*; // Instances, Attribute, Version
import weka.core.converters.*; // ArffLoader, CSVLoader, ArffSaver;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

/**
	@author Jeremy Yang
*/
public class weka_utils
{

  public static Instances LoadCSV(File f)
	throws Exception
  {
    CSVLoader loader=new CSVLoader();
    loader.setSource(f);
    Instances dataset=loader.getDataSet();
    dataset.setClassIndex(dataset.numAttributes()-1); //classifier field 
    //dataset.setClass(attr_class); //classifier field
    return dataset;
  }

  public static Instances LoadARFF(File f)
	throws Exception
  {
    ArffLoader arffloader = new ArffLoader();
    arffloader.setFile(f);
    Instances dataset= arffloader.getDataSet();
    dataset.setClassIndex(dataset.numAttributes()-1); //classifier field
    return dataset;
  }

  public static Boolean WriteARFF(Instances dataset,File f)
	throws Exception
  {
    ArffSaver saver = new ArffSaver();
    saver.setInstances(dataset);
    saver.setFile(f);
    saver.writeBatch();  // Write data in ARFF format.
    return true;
  }

  public static Boolean WriteCSV(Instances dataset,File f)
	throws Exception
  {
    CSVSaver saver = new CSVSaver();
    saver.setInstances(dataset);
    saver.setFile(f);
    saver.writeBatch();  // Write data in CSV format.
    return true;
  }

  public static void DescribeDataset(Instances dataset)
	throws Exception
  {
    System.err.println("Dataset instances: "+dataset.numInstances());
    System.err.println("Dataset classes: "+dataset.numClasses());
    System.err.println("Dataset attributes ("+dataset.numAttributes()+"):");
    for (int j=0;j<dataset.numAttributes();++j)
    {
      Attribute attr=dataset.attribute(j);
      AttributeStats astats=dataset.attributeStats(j);
      System.err.print(String.format("\t%2d. (%s) %s ; ",j+1,Attribute.typeToString(attr),attr.toString()));
      System.err.println(String.format("total=%d ; distinct=%d ; missing=%d",astats.totalCount,astats.distinctCount,astats.missingCount));
    }
  }

  public static RandomForest RandomForest_build(Instances tr_dataset)
	throws Exception
  {
    RandomForest rdfr = new RandomForest();
    rdfr.buildClassifier(tr_dataset);
    //rdfr.setNumTrees(100); //Ok in weka 3.6, not 3.8.
    rdfr.setSeed(1);
    return rdfr; 
  }

  public static Evaluation RandomForest_Evaluation(RandomForest rdfr, Instances dataset_train, Instances dataset_eval)
	throws Exception
  {
    Evaluation eval = new Evaluation(dataset_train);
    // Random rand = new Random(1);  // using seed = 1
    // int folds = 10;
    // eval.crossValidateModel(rdfr, dataset_train, folds, rand);
    eval.evaluateModel(rdfr,dataset_eval); // Evaluation on the test dataset.
    return eval;
  }

  public static String RandomForest_Eval2Txt(Evaluation eval)
	throws Exception
  {
    String outstr="";
    boolean printComplexityStatistics=true;
    String title="";
    outstr+=(eval.toSummaryString(title,printComplexityStatistics)+"\n");
    outstr+=(eval.toMatrixString()+"\n");
    outstr+=(eval.toClassDetailsString()+"\n");

    return outstr;
  }

  /////////////////////////////////////////////////////////////////////////////
  private static int verbose=0;
  private static String ifile=null;
  private static String ofile=null;
  private static boolean arff2csv=false;
  private static boolean csv2arff=false;
  private static void Help(String msg)
  {
    System.err.println(msg+"\n"
      +"weka_utils - weka utility\n"
      +"\n"
      +"usage: weka_utils [options]\n"
      +"input:\n"
      +"  -i IFILE ................. dataset file (CSV|ARFF)\n"
      +"operations:\n"
      +"  -arff2csv .................... convert ARFF to CSV\n"
      +"  -csv2arff .................... convert CSV to ARFF\n"
      +"options:\n"
      +"  -o OFILE ...................... normally CSV\n"
      +"  -v[v[v]] ...................... verbose [very [very]]\n"
      +"  -h ............................ this help\n"
        );
    System.exit(1);
  }
  /////////////////////////////////////////////////////////////////////////////
  private static void ParseCommand(String args[])
  {
    if (args.length==0) Help("");
    for (int i=0;i<args.length;++i)
    {
      if (args[i].equals("-i")) ifile=args[++i];
      else if (args[i].equals("-o")) ofile=args[++i];
      else if (args[i].equals("-arff2csv")) arff2csv=true;
      else if (args[i].equals("-csv2arff")) csv2arff=true;
      else if (args[i].equals("-v")) verbose=1;
      else if (args[i].equals("-vv")) verbose=2;
      else Help("Unknown option: "+args[i]);
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  public static void main(String[] args) throws Exception
  {
    ParseCommand(args);

    File fin = new File(ifile);

    File fout = new File(ofile);

    if (verbose>0) System.err.println("WEKA_Version: "+(new Version()).toString());

    if (arff2csv)
    {
      if (verbose>0) System.err.println("Loading dataset from: "+fin.getAbsolutePath());
      Instances dataset = LoadARFF(fin);
      if (verbose>0) DescribeDataset(dataset);
      if (verbose>0) System.err.println("Writing dataset to: "+fout.getAbsolutePath());
      boolean ok = WriteCSV(dataset,fout);
    }
    else if (csv2arff)
    {
      if (verbose>0) System.err.println("Loading dataset from: "+fin.getAbsolutePath());
      Instances dataset = LoadCSV(fin);
      if (verbose>0) DescribeDataset(dataset);
      if (verbose>0) System.err.println("Writing dataset to: "+fout.getAbsolutePath());
      boolean ok = WriteARFF(dataset,fout);
    }
    else
    {
      Help("ERROR: no operation specified.");
    }
  }
}
