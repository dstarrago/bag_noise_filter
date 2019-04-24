/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataGenerator;

import java.util.ArrayList;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.DenseInstance;
import Utils.StablePropositionalToMultiInstance;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.filters.Filter;

/**
 * Class for generation of artificial MIL data sets.
 * 
 * Generated data sets  have two class of bags: positive and negative. Classes 
 * are balanced, i.e., they roughly have the same number of positive and negative 
 * bags. 
 * 
 * @author Danel
 */
public abstract class Generator {
  
  /**
   * Number of dimensions of the instance space.
   */
  private final int numDimensions;
  
  /**
   * Mean number of instances per bag.
   */
  private final double meanNumInstances;
  
  /**
   * Standard deviation of the number of instance per bag.
   */
  private final double stdDevNumInstances;
  
  /**
   * Number of bags.
   */
  private final int numBags;
  
  /**
   * Array of positive concepts.
   */
  protected final ArrayList<Concept> posConcepts;
  
  /**
   * Array of negative concepts.
   */
  protected final ArrayList<Concept> negConcepts;
  
  /**
   * Noise flags of artificially generated instances. Each instance has an entry 
   * in the array. Since the objective is to identify negative instances as they 
   * were noise, an instance is flagged True if it comes from a negative concept, 
   * which is considered noise. An instance is flagged False if it comes from a 
   * positive concept.
   */
  private ArrayList<Boolean> InstanceFlagList;
  
  /**
   * Random numbers generator.
   */
  protected final Random random;

  public Generator(int numDimensions,
                   double meanNumInstances,
                   double stdDevNumInstances,
                   int numBags,
                   ArrayList<Concept> posConcepts,
                   ArrayList<Concept> negConcepts) throws Exception {
    this.numDimensions = numDimensions;
    this.meanNumInstances = meanNumInstances;
    this.stdDevNumInstances = stdDevNumInstances;
    this.numBags = numBags;
    this.posConcepts = posConcepts;
    this.negConcepts = negConcepts;
    checkInputs(true);
    random = new Random();
  }
  
  public Generator(int numDimensions,
                   double meanNumInstances,
                   double stdDevNumInstances,
                   int numBags) throws Exception {
    this.numDimensions = numDimensions;
    this.meanNumInstances = meanNumInstances;
    this.stdDevNumInstances = stdDevNumInstances;
    this.numBags = numBags;
    checkInputs(false);
    this.posConcepts = new ArrayList<>();
    this.negConcepts = new ArrayList<>();
    random = new Random();
  }
  
  public Generator(double meanNumInstances,
                   double stdDevNumInstances,
                   int numBags) throws Exception {
    this(2, meanNumInstances, stdDevNumInstances, numBags);
  }
  
  private void checkInputs(boolean suppliedConceptLists) throws Exception {
    if (numDimensions <= 0) {
      throw new Exception("Dimensions number can not be least than one");
    }
    if (meanNumInstances < 1) {
      throw new Exception("Mean number of instances per bag can not be least than one");
    }
    if (stdDevNumInstances < 0) {
      throw new Exception("Standard deviation of the number of instance per bag can not be negative");
    }
    if (numBags < 1) {
      throw new Exception("The number of bags can not be least than one");
    }
    if (suppliedConceptLists) {
      if (posConcepts == null || posConcepts.isEmpty()) {
        throw new Exception("No positive concepts have been defined");
      }
      if (negConcepts == null || negConcepts.isEmpty()) {
        throw new Exception("No negative concepts have been defined");
      }
      for (Concept posConcept : posConcepts) {
        if (posConcept.numDimensions() != numDimensions) {
          throw new Exception("Dimension mismatch in positive concept");
        }
      }
      for (Concept negConcept : negConcepts) {
        if (negConcept.numDimensions() != numDimensions) {
          throw new Exception("Dimension mismatch in negative concept");
        }
      }
    }
  }

  public Instances generate() throws Exception {
    if (posConcepts.isEmpty()) {
      throw new Exception("No positive concepts have been defined");
    }
    if (negConcepts.isEmpty()) {
      throw new Exception("No negative concepts have been defined");
    }
    Instances miData = null;
    InstanceFlagList = new ArrayList<>();
    Instances siData = getDatasetStructure();
    int i = 0;
    int half = numBags / 2;
    while (i < half) {
      siData.addAll(getBagInstances(i, getPosInstanceArrays(), 1));
      i++;
    }
    while (i < numBags) {
      siData.addAll(getBagInstances(i, getNegInstanceArrays(), 0));
      i++;
    }
    StablePropositionalToMultiInstance si2mi = new StablePropositionalToMultiInstance();
    /**
     * Data set name
     */
    String dataName = String.format("%s for %d bags with %2.0f ± %2.0f instances/bag,"
            + " %d positive and %d negative concepts", 
            getClass().getSimpleName(), numBags, meanNumInstances, stdDevNumInstances, 
            posConcepts.size(), negConcepts.size());
            
    try {
      si2mi.setInputFormat(siData);
      miData = Filter.useFilter(siData, si2mi);
      miData.setRelationName(dataName);
    } catch (Exception ex) {
      Logger.getLogger(Generator.class.getName()).log(Level.SEVERE, null, ex);
    }
    return miData;
  }
  
  private ArrayList<Instance> getBagInstances(double bagID, 
          ArrayList<double[]> instArrays, double classIndex) {
    ArrayList<Instance> instances = new ArrayList<>();
    for (int i = 0; i < instArrays.size(); i++) {
      double[] vals = new double[numDimensions + 2];
      vals[0] = bagID;
      System.arraycopy(instArrays.get(i), 0, vals, 1, numDimensions);
      vals[numDimensions + 1] = classIndex;
      instances.add(new DenseInstance(1, vals));
    }
    return instances;
  }
  
  private Instances getDatasetStructure() {
    ArrayList<Attribute> attrs = new ArrayList<>();
    /**
     * Bag ID attribute
     */
    ArrayList<String> ids = new ArrayList<>();
    for (int i = 0; i < numBags; i++) {
      ids.add("bag_" + i);
    }
    attrs.add(new Attribute("bagID", ids));
    /**
     * Relational attributes
     */
    for (int i = 0; i < numDimensions; i++) {
      attrs.add(new Attribute("att_" + String.valueOf(i)));
    }
    /**
     * Class attribute
     */
    ArrayList<String> classLabels = new ArrayList<>();
    classLabels.add("neg");
    classLabels.add("pos");
    attrs.add(new Attribute("class", classLabels));
    Instances header = new Instances("noname", attrs, numBags);
    header.setClassIndex(header.numAttributes() - 1);
    return header;
  }
  
  public abstract ArrayList<double[]> getPosInstanceArrays();

  public abstract ArrayList<double[]> getNegInstanceArrays();
  
  protected int genNumBagInstances() {
    double numBagInstances = meanNumInstances + stdDevNumInstances * random.nextGaussian();
    if (numBagInstances < 1) {
      numBagInstances = 1;
    }
    return (int)numBagInstances;
  }
  
  protected void registerPosInstance() {
    InstanceFlagList.add(false);
  }
  
  protected void registerNegInstance() {
    InstanceFlagList.add(true);
  }
  
  public void addPosConcept(double... coords) throws Exception {
   if (coords.length != numDimensions) {
     throw new Exception("Dimension mismatch in positive concept");
   }
   posConcepts.add(new Concept(coords));
  }
  
  public void addNegConcept(double... coords) throws Exception {
   if (coords.length != numDimensions) {
     throw new Exception("Dimension mismatch in negative concept");
   }
   negConcepts.add(new Concept(coords));
  }
  
  public Boolean[] getInstanceFlags() {
    return InstanceFlagList.toArray(new Boolean[0]);
  }
  
}
