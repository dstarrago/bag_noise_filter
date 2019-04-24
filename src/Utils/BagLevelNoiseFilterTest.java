/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import MIFilters.bag.AbstractBagNoiseFilter;
import MIFilters.bag.MinMaxMappingBagNoiseFilter;
import MIFilters.bag.StratifiedAveMappingBagNoiseFilter;
import SIFilters.IPFNoiseFilter;
import SIFilters.KeelNoiseFilter;
//import SIFilters.RNGNoiseFilter;
import java.io.File;
import java.io.FilenameFilter;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
import weka.classifiers.mi.SimpleMI;
import weka.classifiers.mi.MITI;
import weka.classifiers.mi.MILR;
import weka.classifiers.mi.CitationKNN;
import weka.classifiers.mi.MIBoost;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import weka.filters.Filter;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import weka.filters.unsupervised.instance.NonSparseToSparse;

//import MIFilters.instance.InstanceNoiseFilterEvaluator;

/**
 *
 * @author Danel
 */
public class BagLevelNoiseFilterTest {
  
  public static final int SPARSE_DATASET_THRESHOLD = 250;
  private File experimentDir;
  private final Random random = new Random();
  
//  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1");
//  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/Prueba chica");
  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Prueba chica");
  private final File outputDir = new File("C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos");
//  private final File inputDataDir = new File("../Data");
//  private final File inputDataDir = new File("../Insuflated Noise 10%");
//  private final File inputDataDir = new File("../Bag Insuflated Noise 5%");
//  private final File outputDir = new File("../Results");
  private final File outputDataDir = new File(outputDir, "Datasets");
  
// *** METHODS FOR FILTERING BENCHMARK DATA ***
  
  public void testFilterOnBenchmarkData(Class<? extends AbstractBagNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) {
    String timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
    experimentDir = new File(outputDataDir, "Exp" + timeStamp);
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "FilterReport-" + formatedDate + ".txt");
    try {
      FileWriter fileWriter = new FileWriter(report);
      fileWriter.write("Date " + formatedDate + "\n");
      filter(inputDataDir.getCanonicalPath(), filterClass, keelNoiseFilter, fileWriter);
      evalClassifiers(experimentDir.getCanonicalPath(), fileWriter);
      fileWriter.close();
    } catch (Exception ex) {
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void filter(String path, 
          Class<? extends AbstractBagNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) throws IOException, Exception {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "FilterReport-" + formatedDate + ".txt");
    FileWriter fileWriter = new FileWriter(report);
    fileWriter.write("Date " + formatedDate + "\n");
    filter(path, filterClass, keelNoiseFilter, fileWriter);
    fileWriter.close();
  }
  
  public void filter(String path, 
          Class<? extends AbstractBagNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter, FileWriter fileWriter) throws IOException, Exception {
    fileWriter.write("Filter Report \n");
    fileWriter.write("Path: " + path + "\n\n");
    File inputDir = new File(path);
    if (!inputDir.exists() || !inputDir.isDirectory()) {
      throw new IOException("The supplied input path either does not exist or not is a valid directory");
    }
    if (!outputDataDir.exists())
      outputDataDir.mkdirs();
    //File experimentDir = new File(outputDataDir, experimentDirectory);
    experimentDir.mkdir();
    if (keelNoiseFilter == null) {
      fileWriter.write("Filter scheme: " + filterClass.getName());
    }
    else {
      fileWriter.write("Filter scheme: " + filterClass.getName() + " + " + keelNoiseFilter.getName());
    }
    fileWriter.write("\n\n");  
    File[] folderList = inputDir.listFiles();
    for (File folder : folderList) {
      if (!folder.isDirectory()) {
        continue;
      }
      fileWriter.write(String.format("%20s ", folder.getName())); 
      NoiseFilterEvaluator eval = new NoiseFilterEvaluator();
      double avgFilteredPercent = 0;
      int count = 0;
      File outFolder = new File(experimentDir, folder.getName());
      outFolder.mkdir();
      File inCVFolder = new File(folder, "5-Folds-CV");
      File outCVFolder = new File(outFolder, inCVFolder.getName());
      outCVFolder.mkdir();
      File[] fileList = inCVFolder.listFiles();
      for (File dataFile : fileList) {
        File outFile = new File(outCVFolder, dataFile.getName());
        String lowerName = dataFile.getName().toLowerCase();
        if (lowerName.endsWith(".ci")) {
          // do nothing
        } else 
        if (lowerName.contains("test")) {
          Files.copy(dataFile.toPath(), outFile.toPath());          
        } else {
          Instances data = DataSource.read(dataFile.getCanonicalPath());
          data.setClassIndex(data.numAttributes() - 1);
          String ciName = dataFile.getName().substring(0, dataFile.getName().lastIndexOf('.')) + ".ci";
          File ciFile = new File(dataFile.getParentFile(), ciName);
          ObjectInputStream objectIn = new ObjectInputStream(new FileInputStream(ciFile));
          Boolean[] instFlags = (Boolean[])objectIn.readObject();
          Constructor<? extends AbstractBagNoiseFilter> cons;
          AbstractBagNoiseFilter noiseFilter;
          if (keelNoiseFilter == null) {
            cons = filterClass.getConstructor(Instances.class);
            noiseFilter = cons.newInstance(data);
          } else {
            cons = filterClass.getConstructor(Instances.class, keelNoiseFilter.getClass());
            noiseFilter = cons.newInstance(data, keelNoiseFilter);
          }
          noiseFilter.apply();
          eval.part(noiseFilter, instFlags);
          Instances filteredData = noiseFilter.fixedDataset();
          DataSink.write(outFile.getCanonicalPath(), filteredData);
          avgFilteredPercent += 100.0 * ((double) noiseFilter.numFiltered() / (double) data.numInstances());
          count ++;
        }
      }
      avgFilteredPercent /= count;
      fileWriter.write(String.format(" average %% of detected noisy bags %4.2f  precision: "
              + "%4.2f  recall: %4.2f \n", avgFilteredPercent, eval.precision(), eval.recall()));  
      fileWriter.flush();
    }
    fileWriter.write("\n");  
  }
  
// *** METHODS FOR EVALUATING BENCHMARK DATA ***
  
  public void evaluateClassifiersOnBenchmarkData() {
    try {
      evalClassifiers(inputDataDir.getCanonicalPath());
    } catch (Exception ex) {
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void evalClassifiers(String path) throws IOException, Exception {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "ClassificationReport-" + formatedDate + ".txt");
    FileWriter fileWriter = new FileWriter(report);
    fileWriter.write("Date " + formatedDate + "\n");
    evalClassifiers(path, fileWriter);
    fileWriter.close();
  }
  
  private void evaluate(Classifier classifier, File inputDir, FileWriter fileWriter) throws IOException, Exception {
    fileWriter.write(String.format("*** %s *** \n\n", classifier.getClass().getName()));
    TrainArffFilter fileFilter = new TrainArffFilter();
    File[] folderList = inputDir.listFiles();
    for (File folder : folderList) {
      if (!folder.isDirectory()) {
        continue;
      }
      File inCVFolder = new File(folder, "5-Folds-CV");
      File[] fileList = inCVFolder.listFiles(fileFilter);
      double meanAccuracy = 0;
      for (File trainFile : fileList) {
        Instances trainData = DataSource.read(trainFile.getCanonicalPath());
        String testName = trainFile.getName().replaceFirst("train", "test");
        File testFile = new File(trainFile.getParentFile(), testName);
        Instances testData = DataSource.read(testFile.getCanonicalPath());
        if (trainData.attribute(1).relation().numAttributes() >= SPARSE_DATASET_THRESHOLD) {
          trainData = toSparseBag(trainData);
          testData = toSparseBag(testData);
        }
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(trainData.numAttributes() - 1);
        classifier.buildClassifier(trainData);
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(classifier, testData);
        meanAccuracy += eval.pctCorrect();
      }
      meanAccuracy /= fileList.length;
      String reportStr = String.format("%20s: %6.4f \n", folder.getName(), meanAccuracy);
      fileWriter.write(reportStr);
      fileWriter.flush();
    }
    fileWriter.write("\n");
  }
  
  public void evalClassifiers(String path, FileWriter fileWriter) throws IOException {
    fileWriter.write("Classification Report \n");
    fileWriter.write("Path: " + path + "\n\n");
    File inputDir = new File(path);
    if (!inputDir.exists() || !inputDir.isDirectory()) {
      throw new IOException("The supplied input path either does not exist or not is a valid directory");
    }
    if (!outputDir.exists())
      outputDir.mkdirs();
    try {
      SimpleMI simpleMI = new SimpleMI();
      simpleMI.setOptions(new String[]{"-M", "2", "-W", "weka.classifiers.trees.J48"});
      evaluate(simpleMI, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador SimpleMI");
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MITI miti = new MITI();
      evaluate(miti, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MITI");
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MILR milr = new MILR();
      evaluate(milr, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MILR");
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      CitationKNN citation = new CitationKNN();
      citation.setOptions(new String[]{"-R", "2", "-C", "2"});
      evaluate(citation, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador CitationKNN");
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MIBoost boost = new MIBoost();
      boost.setOptions(new String[]{"-R", "10", "-B", "0", "-W", "weka.classifiers.bayes.NaiveBayes"});
      evaluate(boost, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MIBoost");
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public Instances toSparseBag(Instances B) throws Exception {
    NonSparseToSparse filter = new NonSparseToSparse();
    Instances compact = new Instances(B);
    for (int i = 0; i < compact.numInstances(); i++) {
      Instances bag = compact.get(i).relationalValue(1);
      filter.setInputFormat(bag);   
      Instances compBag = Filter.useFilter(bag, filter);
      bag.clear();
      bag.addAll(compBag);
    }
    return compact;
  }
  
  private class ArffFilter implements FilenameFilter {

    @Override
    public boolean accept(File dir, String name) {
      String lowerName = name.toLowerCase();
      return lowerName.endsWith(".arff");
    }
    
  }
  
  private class TrainArffFilter implements FilenameFilter {

    @Override
    public boolean accept(File dir, String name) {
      String lowerName = name.toLowerCase();
      return lowerName.contains("train") && lowerName.endsWith(".arff");
    }
    
  }
  
  public void buildCheckFiles() {
    String objectPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Bag Insuflated Noise 0%";
    String subjectPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Bag Insuflated Noise 20%";
    File objectDir = new File(objectPath);
    File subjectDir = new File(subjectPath);
    File[] objectFolderList = objectDir.listFiles();
    for (File objectFolder : objectFolderList) {
      if (!objectFolder.isDirectory()) {
        continue;
      }
      File subjectFolder = new File(subjectDir, objectFolder.getName());
      File objectCVFolder = new File(objectFolder, "5-Folds-CV");
      File subjectCVFolder = new File(subjectFolder, objectCVFolder.getName());
      File[] objectFileList = objectCVFolder.listFiles();
      for (File objectFile : objectFileList) {
          File subjectFile = new File(subjectCVFolder, objectFile.getName());
          String objectLowerCaseName = objectFile.getName().toLowerCase();
          if (!objectLowerCaseName.contains("test")) {
            try {
              Instances object = DataSource.read(objectFile.getCanonicalPath());
              object.setClassIndex(object.numAttributes() - 1);
              Instances subject = DataSource.read(subjectFile.getCanonicalPath());
              subject.setClassIndex(subject.numAttributes() - 1);
              Boolean[] diff = checkDifferences(object, subject);
              String checkFileName = subjectFile.getName().substring(0,
                      subjectFile.getName().lastIndexOf('.')) + ".ci";
              File checkFile = new File(subjectCVFolder, checkFileName);
              try (ObjectOutputStream objectOut = new ObjectOutputStream(new FileOutputStream(checkFile))) {
                objectOut.writeObject(diff);
              }
            } catch (Exception ex) {
              Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
            }
          } 
      }
    }
  }
  
  private Boolean[] checkDifferences(Instances A, Instances B) {
    Boolean[] diff = new Boolean[A.numInstances()];
    for (int i = 0; i < diff.length; i++) {
      diff[i] = A.get(i).classValue() != B.get(i).classValue();
    }
    return diff;
  }
  
  private void testCheckFile() {
    String path = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos"
            + "/Bag Insuflated Noise 5%/01 Musk1/5-Folds-CV/Musk1-f1-train.ci";
    File targetFile = new File(path);
    try {
      ObjectInputStream objectIn = new ObjectInputStream(new FileInputStream(targetFile));
      Boolean[] instFlags = (Boolean[])objectIn.readObject();
      for (Boolean instFlag : instFlags) {
        System.out.println(instFlag);
      }
    } catch (Exception ex) {
      Logger.getLogger(BagLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    BagLevelNoiseFilterTest test = new BagLevelNoiseFilterTest();
//    test.evaluateClassifiersOnBenchmarkData();
    test.testFilterOnBenchmarkData(StratifiedAveMappingBagNoiseFilter.class, IPFNoiseFilter.class);
}

}

