package edu.indiana.sice.idsl.weka;

import java.io.*;
import java.util.*;

import weka.core.*; // Instances, Attribute
import weka.core.converters.*; // ArffLoader, CSVLoader, ArffSaver;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import chemaxon.util.MolHandler;
import chemaxon.sss.search.MolSearch;
import chemaxon.struc.Molecule;
import chemaxon.formats.*; // MolImporter, MolFormatException
import chemaxon.sss.search.SearchException;

/**	Extends WEKA Instances class to handle ChemAxon molecule sets.
	SMILES and molnames stored as Attributes.
	Problem: smiles with backslash are returned by toString() incorrectly (single quoted?).
	Likewise names with spaces.

	JOELib MolInstance[s] classes are an alternative approach, but
	require modifying WEKA source.

	Attributes can be calculated properties.

	@author Jeremy Yang
*/
public class MolecularDataset extends weka.core.Instances
{

  ////////////////////////////////////////////////////////////////////////////////
  public MolecularDataset(String name,FastVector attrs,int size) //constructor override
  {
    super(name,attrs,size);
  }

  ////////////////////////////////////////////////////////////////////////////////
  public int loadMoleculeFile(File ifile)
	throws IOException
  {
    MolImporter molReader=new MolImporter(new FileInputStream(ifile));

    String fmtdesc=MFileFormatUtil.getFormat(molReader.getFormat()).getDescription();
    System.err.println("DEBUG: input format:  "+molReader.getFormat()+" ("+fmtdesc+")");

    int n_mols_in=0;
    int n_mols_loaded=0;
    Molecule mol;
    int n_failed=0;
    while (true)
    {
      try { mol=molReader.read(); }
      catch (MolFormatException e)
      {
        System.err.println("DEBUG: MolImporter failed: "+e.getMessage());
        ++n_failed;
        continue;
      }
      if (mol==null) break;
      ++n_mols_in;

      String smi;
      try { smi = MolExporter.exportToFormat(mol,"smiles:-a"); } //Kekule-smiles
      catch (MolFormatException e)
      {
        System.err.println("DEBUG: MolExporter failed: "+e.getMessage());
        ++n_failed;
        continue;
      }

      if (n_mols_loaded==0)
      {
        FastVector attrs = new FastVector(0);
        attrs.addElement(new Attribute("smiles",(FastVector) null));
        attrs.addElement(new Attribute("name",(FastVector) null));
        Instance inst = new DenseInstance(attrs.size());
        inst.setDataset(this);
        this.add(inst);
      }
      else
      {
        this.add(new DenseInstance(this.lastInstance()));
      }
      this.lastInstance().setValue(this.attribute("smiles"),smi);
      this.lastInstance().setValue(this.attribute("name"),mol.getName());
      ++n_mols_loaded;
    }
    return n_mols_in;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /**	For testing.
  */
  public static void ShowMoldata(MolecularDataset moldataset)
  {
    System.err.println("numAttributes() = "+moldataset.numAttributes());
    System.err.println("numInstances() = "+moldataset.numInstances());
    for (int i=0;i<moldataset.numInstances();++i)
    {
      System.err.println(""+i+".");
      for (int j=0;j<moldataset.instance(i).numAttributes();++j)
      {
        System.err.println("\t"+j+". "+moldataset.attribute(j).name()+": "+moldataset.instance(i).toString(moldataset.instance(i).attribute(j)));
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /**	For testing.
  */
  public static void main(String [] args)
  {
    if (args.length!=1)
    {
      System.err.println("usage: progname <infile>");
      System.exit(1);
    }

    MolecularDataset moldataset = null;
    int n_mols = 0;
    try {
      FastVector attrs = new FastVector(0);
      attrs.addElement(new Attribute("smiles",(FastVector) null)); //allows String value
      attrs.addElement(new Attribute("name",(FastVector) null)); //allows String value
      moldataset = new MolecularDataset(args[0],attrs,0);
      n_mols = moldataset.loadMoleculeFile(new File(args[0]));
    } catch (IOException e) {
      System.err.println("ERROR: "+args[0]+": "+e.getMessage());
    }
    System.err.println("n_mols = "+n_mols);

    ShowMoldata(moldataset);

    // Now add Attribute "mwt".
    Molecule mol ;
    moldataset.insertAttributeAt(new Attribute("mwt"),moldataset.numAttributes());
    for (int i=0;i<moldataset.numInstances();++i)
    {
      String smi=moldataset.instance(i).toString(moldataset.attribute("smiles")).replaceAll("'","");
      try { mol=MolImporter.importMol(smi,"smiles:"); }
      catch (MolFormatException e)
      {
        System.err.println("ERROR: MolImporter failed: "+e.getMessage());
        continue;
      }
      MolHandler mhand = new MolHandler();
      mhand.setMolecule(mol);
      Float mwt=mhand.calcMolWeight();
      int heavycount=mhand.getHeavyAtomCount();
      moldataset.instance(i).setValue(moldataset.attribute("mwt"),mwt);
    }

    ShowMoldata(moldataset);
  }
}
