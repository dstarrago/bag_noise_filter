/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SIFilters;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import keel.Dataset.Attributes;
import keel.Dataset.DatasetException;
import keel.Dataset.HeaderFormatException;
import keel.Dataset.Instance;
import keel.Dataset.InstanceSet;
import org.core.Fichero;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Abstract class used as an interface to noise filter classes of the KEEL platform.
 * 
 * The abstract method runNoiseFilter has to be defined by descendent classes.
 * 
 * The followings methods should be invoked by descendent classes:
 * - setDecisions, to inform the decision of the filter on each instance (is/isn't noisy)
 * - setProposedClasses, to inform the class label proposed by the filter to each instance.
 * - setNumNoisyExamples, to inform the number of noisy instance detected by the filter.
 * 
 * @author Danel
 */
public abstract class KeelNoiseFilter {
  
  /**
   * Path where temporary files will be located.
   */
  private static final String TMP_FOLDER = System.getProperty("java.io.tmpdir") + "/";
  
  /**
   * Time mark used to identify with a unique name temporary files.
   */
  protected String timeStamp;

  /**
   * To generate a random number to add to the timeStamp.
   */
  private Random random = new Random();
  
  /**
   * Location and name of auxiliary training data;
   */
  private final String traENN = TMP_FOLDER + "aux_tra.dat";
  
  /**
   * Location and name of auxiliary test data. 
   */
  private final String tstENN = TMP_FOLDER + "aux_tst.dat";
  
  /**
   * Location and name of the configuration file for ENN algorithm. (Consider to remove)
   */
  private static final String CONFIG_FILENAME = TMP_FOLDER + "config_ENN.txt";
  
  /**
   * Number of nearest neighbors used by ENN algorithm by default. (Consider to remove)
   */
  private final int defaultKNN = 3;
  
  /**
   * Location and name of the temporary file used by KEEL to run the filter algorithm.
   * The file contains the data set that is going to be filtered.
   */
  private String dataFileName;
  
  /**
   * Memory representation of the data set that is going to be filtered.
   */
  private Instances dataset;
  
  /**
   * Array of decisions of the noise filter over instances of the data set. 
   * 
   * A value of 'True' means that the corresponding instance has 
   * been identified as noisy by the filter. 
   */
  private boolean[] decisions;
  
  /**
   * Array of proposed classes for the instances of the data set. 
   * 
   * The value proposedClasses[i] is the index of class label 
   * that the filter has determined that best suits to the ith instance.
   */
  private int[] proposedClasses;
  
  /**
   * Number of examples that have been identified as noisy by the filter.
   */
  private int numNoisyExamples;

  /**
   * Constructor for the abstract filter that operates over a representation 
   * of the data set that is stored in memory. 
   * 
   * @param dataset set of instances where the filter will be applied. 
   * @throws Exception 
   */
  public KeelNoiseFilter(Instances dataset) throws Exception {
    if (dataset == null)
      throw new Exception("Non dataset has been suplied for noise filtering");
    this.dataset = new Instances(dataset);
    dataset.setClassIndex(dataset.numAttributes() - 1);
    timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
    String tmpDataFileName = String.format("%sData-%s.arff", TMP_FOLDER, timeStamp);
    try {
      ConverterUtils.DataSink.write(tmpDataFileName, dataset);
    } catch (Exception ex) {
      Logger.getLogger(KeelNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    this.dataFileName = tmpDataFileName;
//    CreateConfigFile(tmpDataFileName, tmpDataFileName, traENN, tstENN, defaultKNN); // (Consider to remove)
    initDataStructures(tmpDataFileName, true);
    run();
    File tmpFile = new File(tmpDataFileName);
    tmpFile.delete();
  }
  
  /**
   * Constructor for the abstract filter that operates over a representation 
   * of the data set that is stored in a file. 
   * 
   * @param dataFileName name of the file containing the set of instances where 
   * the filter will be applied. 
   * 
   * @throws Exception 
   */
  public KeelNoiseFilter(String dataFileName) throws Exception {
    dataset = ConverterUtils.DataSource.read(dataFileName);
    dataset.setClassIndex(dataset.numAttributes() - 1);
    this.dataFileName = dataFileName;
    CreateConfigFile(dataFileName, dataFileName, traENN, tstENN, defaultKNN);		
    initDataStructures(dataFileName, true);
    run();
  }
  
  /**
   * Initialize data structures used by KEEL algorithms. 
   * 
   * @param trainingSet name of the file containing the set of instances where 
   * the filter will be applied.
   * 
   * @param loadAttributes a "given" KEEL attribute.
   */
  private void initDataStructures(String trainingSet, boolean loadAttributes){
    InstanceSet is = new InstanceSet();
    Instance[] instancesTrain;	// all the instances of the training set
    if(loadAttributes)
      Attributes.clearAll();
    try {	
      is.readSet(trainingSet, loadAttributes);
    }catch(DatasetException e){
      System.exit(1);
    } catch (HeaderFormatException e) {
      System.exit(1);
      }
    instancesTrain = is.getInstances();
    Parameters.numClasses = Attributes.getOutputAttribute(0).getNumNominalValues();
    Parameters.numAttributes = Attributes.getInputAttributes().length;
    Parameters.numInstances = instancesTrain.length;
  }

  /**
   * Makes some preprocessing and executes the filtering algorithm.
   * 
   * @throws Exception 
   */
  private void run() throws Exception {
    boolean[] cleanExample = new boolean[Parameters.numInstances];
    Arrays.fill(cleanExample, true);
    runNoiseFilter(cleanExample);
  }
  
  /**
   * Execute the filtering algorithm. This method is abstract and has to be 
   * defined in specialized classes.
   * 
   * @param cleanExample a "given" KEEL parameter.
   */
  protected abstract void runNoiseFilter(boolean[] cleanExample); 
  
  /**
   * Sets the decisions of the filter. 
   * 
   * This method should be used by any specialized class to set its decisions
   * after the filtering process is finished.
   * 
   * @param decisions array of boolean decisions.
   */
  protected void setDecisions(boolean[] decisions) {
    this.decisions = decisions;
  }
  
  /**
   * Sets the class labels proposed by the filter. 
   * 
   * This method should be used by any specialized class after the filtering 
   * process is finished to set the proposed instance class labels.
   * 
   * @param proposedClasses array of proposed instance class labels.
   */
  protected void setProposedClasses(int[] proposedClasses) {
    this.proposedClasses = proposedClasses;
  }
  
  /**
   * Gets the name of the file containing the data set on which the filter 
   * algorithm is applied.
   * 
   * @return an string representing the location and name of the file.
   */
  protected String dataFileName() {
    return dataFileName;
  }
  
  /**
   * Sets the number of examples identified as noise by the filter algorithm.
   * 
   * This method should be used by any specialized class after the filtering 
   * process is finished.
   * 
   * @param numNoisyExamples number of noisy examples.
   */
  protected void setNumNoisyExamples(int numNoisyExamples) {
    this.numNoisyExamples = numNoisyExamples;
  }

  /**
   * Gets the number of examples identified as noise by the filter algorithm.
   * 
   * @return number of noisy examples.
   */  
  public int numNoisyExamples() {
    return numNoisyExamples;  
  }
  
  /**
   * Gets the number of examples identified as safe (i.e., not noisy) by the 
   * filter algorithm.
   * 
   * @return number of safe examples.
   */
  public int numSafeExamples() {
    return dataset.numInstances() - numNoisyExamples;
  }

  /**
   * Gets the set of examples identified as noise by the filter algorithm.
   * 
   * @return a data set containing the detected noisy examples.
   */  
  public Instances noisyExamples() {
    Instances noisyExamples = new Instances(dataset, 0);
    for (int i = 0; i < dataset.numInstances(); i++) {
      if (decisions[i]) {
        noisyExamples.add(dataset.get(i));
      }
    }
    return noisyExamples;
  }

  /**
   * Gets the set of examples identified as safe (i.e., not noisy) by the 
   * filter algorithm.
   * 
   * @return a data set containing the detected safe examples.
   */  
  public Instances safeExamples() {
    Instances safeExamples = new Instances(dataset, 0);
    for (int i = 0; i < dataset.numInstances(); i++) {
      if (!decisions[i]) {
        safeExamples.add(dataset.get(i));
      }
    }
    return safeExamples;
  }
  
  /**
   * Prints the set of examples identified as noise by the filter algorithm.
   * 
   * @return a String representing the detected noisy examples. Each example in 
   * a row.
   */
  public String noisyExamplesToString() {
    StringBuilder noisyExamples = new StringBuilder();
    for (int i = 0; i < dataset.numInstances(); i++) {
      if (decisions[i]) {
        noisyExamples.append(dataset.get(i).toString()).append("\n");
      }
    }
    return noisyExamples.toString();
  }

  /**
   * Prints the set of examples identified as safe (i.e., not noisy) by the 
   * filter algorithm.
   * 
   * @return a String representing the detected safe examples. Each example in 
   * a row.
   */  
  public String safeExamplesToString() {
    StringBuilder safeExamples = new StringBuilder();
    for (int i = 0; i < dataset.numInstances(); i++) {
      if (!decisions[i]) {
        safeExamples.append(dataset.get(i).toString()).append("\n");
      }
    }
    return safeExamples.toString();
  }

  /**
   * Prints the example and proposed class label for those examples that has 
   * been identified as noise by the filter algorithm.
   * 
   * @return a String with a list of examples with its proposed class label. 
   * Each example in a row.
   */
  public String fixedClassesToString() {
    StringBuilder fixedClasses = new StringBuilder();
    for (int i = 0; i < dataset.numInstances(); i++) {
      if (decisions[i]) {
        String newAttrValue = dataset.classAttribute().value(proposedClasses[i]);
        fixedClasses.append(dataset.get(i).toString()).append(" --> ").
          append(newAttrValue).append("\n");
      }
    }
    return fixedClasses.toString();
  }
  
  /**
   * Gets the decisions of this noise filter.
   * 
   * @return an array with the decisions for all the instances in the data set. 
   */
  public boolean[] decisions() {
    return java.util.Arrays.copyOf(decisions, decisions.length);
  }

  /**
   * Gets the number of examples where the filter has been applied.
   * 
   * @return number of examples.
   */  
  public int numExamples() {
    return dataset.numInstances();
  }

  /**
   * Gets a given example from the data set where the filter has been applied.
   * 
   * @param index index of the requested example.
   * 
   * @return the requested example.
   */  
  public weka.core.Instance getExample(int index) {
    return dataset.get(index);
  }

  /**
   * Gets the class label of an instance in the original data set.
   * 
   * @param index index of the instance.
   * 
   * @return index of the class label. 
   */  
  public double getClass(int index) {
    return dataset.get(index).classValue();
  }

  /**
   * Informs if a given instance has been identified as noise by the filter.
   * 
   * @param index index of the instance.
   * 
   * @return true if the instance has been identified as noise by the filter, 
   * false otherwise.
   * 
   */  
  public boolean isNoisy(int index) {
    return decisions[index];
  }

  /**
   * Informs the class label proposed by the filter for a given instance. 
   * 
   * @param index index of the instance.
   * 
   * @return index of the class label proposed by the filter.
   */  
  public double proposedClass(int index) {
    return proposedClasses[index];
  }

  /**
   * Method used to create the configuration file for ENN algorithm. 
   * (Consider to remove)
   * 
   * @param tra_in
   * @param tst_in
   * @param tra_out
   * @param tst_out
   * @param k_value
   * @throws Exception 
   */  
  private void CreateConfigFile(String tra_in, String tst_in, String tra_out, String tst_out, int k_value) throws Exception{
      String content = "";
      content += "algorithm = Edited Nearest Neighbor";
      content += "\ninputData = \"" + tra_in + "\" \"" + tst_in +"\"";
      content += "\noutputData = \"" + tra_out + "\" \"" + tst_out + "\"";
      content += "\n\nNumber of Neighbors = " + k_value;
      content += "\nDistance Function = Euclidean";
      Fichero.escribeFichero(CONFIG_FILENAME, content);
    }	

}
