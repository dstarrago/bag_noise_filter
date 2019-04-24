/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import MIFilters.instance.AbstractInstanceNoiseFilter;
import MIFilters.instance.IterativeNNInstanceNoiseMultiFilter;
import SIFilters.KeelNoiseFilter;
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
import java.io.ObjectInputStream;
import java.util.Random;
import weka.filters.unsupervised.instance.NonSparseToSparse;
//import DataGenerator.BagLevelNoiseInsufflator;

//import MIFilters.instance.InstanceNoiseFilterEvaluator;

/**
 *
 * @author Danel
 */
public class InstanceLevelNoiseFilterTest {
  
  private final int numFolds = 5;
  private FileWriter log;
  private final int numClassifers = 5;
  private final int ind_simpleMI = 0;
  private final int ind_MITI = 1;
  private final int ind_MILR = 2;
  private final int ind_CitationKNN = 3;
  private final int ind_MIBoost = 4;
  public static final int SPARSE_DATASET_THRESHOLD = 250;
  private File experimentDir;
  private final Random random = new Random();
  
//  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1");
//  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/Prueba chica");
////  private final File inputDataDir = new File("C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Prueba chica");
////  private final File outputDir = new File("C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos");
//  private final File inputDataDir = new File("../Data");
//  private final File inputDataDir = new File("../IN 20%");
  private final File inputDataDir = new File("/home/danels/Insuflated Noise 20%");
//  private final File inputDataDir = new File("../NN-IPF Help/Exp1483631564457-804183559");
  private final File outputDir = new File("../Results");
  private final File outputDataDir = new File(outputDir, "Datasets");
  
//  private final String[] dataSets = new String[] {"63 Corel01vs02", "64 Corel01vs03", 
//      "65 Corel01vs04", "66 Corel01vs05", "67 Corel02vs03", "68 Corel02vs04", 
//      "69 Corel02vs05", "70 Corel03vs04", "71 Corel03vs05", "72 Corel04vs05", 
//      "03 Atoms", "04 Bonds", "05 Chains",
//      "08 Elephant", "09 Fox", "10 Tiger", 
//      "01 Musk1","02 Musk2", 
//      "19 TREC9-1", "20 TREC9-2", "21 TREC9-3", "22 TREC9-4", 
//      "23 TREC9-7", "24 TREC9-9", "25 TREC9-10", 
//      "32 WIR7", "33 WIR8", "34 WIR9"};
  
// *** METHODS FOR FILTERING AND EVALUATING ARTIFICIAL DATA ***
  
  private double evaluate(Classifier classifier, Instances dataset) {
    double meanAccuracy = 0;
    dataset.stratify(numFolds);
    for (int i = 0; i < numFolds; i++) {
      Instances train = dataset.trainCV(numFolds, i);
      Instances test = dataset.testCV(numFolds, i);
      try {
        classifier.buildClassifier(train);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(classifier, test);
        meanAccuracy += eval.pctCorrect();
      } catch (Exception ex) {
        Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
      }
    }
    meanAccuracy /= numFolds;
    return meanAccuracy;
  }
  
  public void classificationOnArtificialData(Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) {
    try {
      Date date = new Date();
      DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss.SSSS");
      if (!outputDir.exists())
        outputDir.mkdirs();
      File outputFile = new File(outputDir, "output-" + df.format(date) + ".txt");
      log = new FileWriter(outputFile);
      if (keelNoiseFilter == null) {
        log.write("Filter scheme: " + filterClass.getName());
      }
      else {
        log.write("Filter scheme: " + filterClass.getName() + " + " + keelNoiseFilter.getName());
      }
      log.write("\n");
      log.write("\n");
      log.flush();
    } catch (IOException ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    evaluateArtificialData(filterClass, keelNoiseFilter);
    try {
      log.close();
    } catch (IOException ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
}
 
  private void evaluateArtificialData(Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) {
    ArffFilter arffFilter = new ArffFilter();
    String adPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Prueba chica";
//    String adPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/ArtificialData";
//    String adPath = "../ArtificialData";
    double[] clAccuracy = new double[numClassifers];
    File adDir = new File(adPath);
    File[] folderList = adDir.listFiles();
    for (File folder : folderList) {
      try {
        log.write("Dataset: " + folder.getName());
        log.write("\n");
        log.write("\n");
        log.flush();
      } catch (IOException ex) {
        Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
      }
      double tp = 0;
      double tn = 0;
      double fp = 0;
      double fn = 0;
      double precision = 0;
      double recall = 0;
      double accuracy = 0; 
      java.util.Arrays.fill(clAccuracy, 0);
      File[] fileList = folder.listFiles(arffFilter);
      for (File dataFile : fileList) {
        try {
          Instances data = DataSource.read(dataFile.getCanonicalPath());
          data.setClassIndex(data.numAttributes() - 1);
          String ciName = dataFile.getName().substring(0, dataFile.getName().lastIndexOf('.')) + ".ci";
          File ciFile = new File(dataFile.getParentFile(), ciName);
          ObjectInputStream objectIn = new ObjectInputStream(new FileInputStream(ciFile));
          Boolean[] instFlags = (Boolean[])objectIn.readObject();
          Constructor<? extends AbstractInstanceNoiseFilter> cons;
          AbstractInstanceNoiseFilter noiseFilter;
          if (keelNoiseFilter == null) {
            cons = filterClass.getConstructor(Instances.class);
            noiseFilter = cons.newInstance(data);
          } else {
            cons = filterClass.getConstructor(Instances.class, keelNoiseFilter.getClass());
            noiseFilter = cons.newInstance(data, keelNoiseFilter);
          }
          noiseFilter.apply();
          NoiseFilterEvaluator eval = new NoiseFilterEvaluator(noiseFilter, instFlags);
          tp += eval.truePositives();
          tn += eval.trueNegatives();
          fp += eval.falsePositives();
          fn += eval.falseNegatives();
          precision += eval.precision();
          recall += eval.recall();
          accuracy += eval.accuracy();
          Instances fData = noiseFilter.safeExamples();
          fData.setClassIndex(fData.numAttributes() - 1);
          evaluate(fData, clAccuracy);
        } catch (Exception ex) {
          Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
        }
      }
      try {
        log.write(String.format("TP: %4.2f \n", tp / fileList.length));
        log.write(String.format("TN: %4.2f \n", tn / fileList.length));
        log.write(String.format("FP: %4.2f \n", fp / fileList.length));
        log.write(String.format("FN: %4.2f \n", fn / fileList.length));
        log.write(String.format("Precision: %4.2f \n", precision / fileList.length));
        log.write(String.format("Recall: %4.2f \n", recall / fileList.length));
        log.write(String.format("Accuracy: %4.2f \n", accuracy / fileList.length));
        log.write("\n");
        log.write(String.format("Classifier: %15s accuracy: %4.2f \n", "SimpleMI", clAccuracy[ind_simpleMI] / fileList.length));
        log.write(String.format("Classifier: %15s accuracy: %4.2f \n", "MITI", clAccuracy[ind_MITI] / fileList.length));
        log.write(String.format("Classifier: %15s accuracy: %4.2f \n", "MILR", clAccuracy[ind_MILR] / fileList.length));
        log.write(String.format("Classifier: %15s accuracy: %4.2f \n", "CitationKNN", clAccuracy[ind_CitationKNN] / fileList.length));
        log.write(String.format("Classifier: %15s accuracy: %4.2f \n", "MIBoost", clAccuracy[ind_MIBoost] / fileList.length));
        log.write("\n");
        log.write("\n");
        log.flush();
      } catch (IOException ex) {
        Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
      }
    }
  }
  
  private void evaluate(Instances dataset, double[] clAccuracy) {
    try {
      SimpleMI simpleMI = new SimpleMI();
      simpleMI.setOptions(new String[]{"-M", "2", "-W", "weka.classifiers.trees.J48"});
      clAccuracy[ind_simpleMI] += evaluate(simpleMI, dataset);
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MITI miti = new MITI();
      clAccuracy[ind_MITI] += evaluate(miti, dataset);
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MILR milr = new MILR();
      clAccuracy[ind_MILR] += evaluate(milr, dataset);
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      CitationKNN citation = new CitationKNN();
      citation.setOptions(new String[]{"-R", "2", "-C", "2"});
      clAccuracy[ind_CitationKNN] += evaluate(citation, dataset);
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MIBoost boost = new MIBoost();
      boost.setOptions(new String[]{"-R", "10", "-B", "0", "-W", "weka.classifiers.bayes.NaiveBayes"});
      clAccuracy[ind_MIBoost] += evaluate(boost, dataset);
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  public void testOnArtificialData(Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) {
    ArffFilter arffFilter = new ArffFilter();
//    String adPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/ArtificialData";
    String adPath = "../ArtificialData";
    File adDir = new File(adPath);
    File[] folderList = adDir.listFiles();
    for (File folder : folderList) {
      try {
        System.out.println("Dataset: " + folder.getCanonicalPath());
      } catch (IOException ex) {
        Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
      }
      double tp = 0;
      double tn = 0;
      double fp = 0;
      double fn = 0;
      double precision = 0;
      double recall = 0;
      double accuracy = 0;
      File[] fileList = folder.listFiles(arffFilter);
      for (File dataFile : fileList) {
        try {
          Instances data = DataSource.read(dataFile.getCanonicalPath());
          data.setClassIndex(data.numAttributes() - 1);
          String ciName = dataFile.getName().substring(0, dataFile.getName().lastIndexOf('.')) + ".ci";
          File ciFile = new File(dataFile.getParentFile(), ciName);
          ObjectInputStream objectIn = new ObjectInputStream(new FileInputStream(ciFile));
          Boolean[] instFlags = (Boolean[])objectIn.readObject();
          Constructor<? extends AbstractInstanceNoiseFilter> cons;
          cons = filterClass.getConstructor(Instances.class, keelNoiseFilter.getClass());
          AbstractInstanceNoiseFilter noiseFilter = cons.newInstance(data, keelNoiseFilter);
          noiseFilter.apply();
          NoiseFilterEvaluator eval = new NoiseFilterEvaluator(noiseFilter, instFlags);
          tp += eval.truePositives();
          tn += eval.trueNegatives();
          fp += eval.falsePositives();
          fn += eval.falseNegatives();
          precision += eval.precision();
          recall += eval.recall();
          accuracy += eval.accuracy();
        } catch (Exception ex) {
          Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
        }
      }
      System.out.println();
      System.out.println(String.format("TP: %4.2f", tp / fileList.length));
      System.out.println(String.format("TN: %4.2f", tn / fileList.length));
      System.out.println(String.format("FP: %4.2f", fp / fileList.length));
      System.out.println(String.format("FN: %4.2f", fn / fileList.length));
      System.out.println(String.format("Precision: %4.2f", precision / fileList.length));
      System.out.println(String.format("Recall: %4.2f", recall / fileList.length));
      System.out.println(String.format("Accuracy: %4.2f", accuracy / fileList.length));
      System.out.println();
      System.out.println();
    }
  }

// *** METHODS FOR FILTERING BENCHMARK DATA ***
  
  public void testFilterOnBenchmarkData(Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) {
    String timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
    experimentDir = new File(outputDataDir, "Exp" + timeStamp);
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss.SSSS");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "FilterReport-" + formatedDate + ".txt");
    try {
      FileWriter fileWriter = new FileWriter(report);
      fileWriter.write("Date " + formatedDate + "\n");
      filter(inputDataDir.getCanonicalPath(), filterClass, keelNoiseFilter, fileWriter);
      evalClassifiers(experimentDir.getCanonicalPath(), fileWriter);
      fileWriter.close();
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void testFilterOnDataset(Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter, String datasetName) {
    String timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
    experimentDir = new File(outputDataDir, "Exp" + timeStamp);
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss.SSSS");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "FilterReport-" + formatedDate + ".txt");
    try {
      FileWriter fileWriter = new FileWriter(report);
      fileWriter.write("Date " + formatedDate + "\n");
      filterDataset(inputDataDir.getCanonicalPath(), filterClass, keelNoiseFilter, 
              fileWriter, datasetName);
      evalClassifiersOnDataset(experimentDir.getCanonicalPath(), fileWriter, 
              datasetName);
      fileWriter.close();
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void filter(String path, 
          Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter) throws IOException, Exception {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss.SSSS");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "FilterReport-" + formatedDate + ".txt");
    FileWriter fileWriter = new FileWriter(report);
    fileWriter.write("Date " + formatedDate + "\n");
    filter(path, filterClass, keelNoiseFilter, fileWriter);
    fileWriter.close();
  }
  
  public void filter(String path, 
          Class<? extends AbstractInstanceNoiseFilter> filterClass,
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
      fileWriter.write(String.format("%30s ", folder.getName())); 
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
        if (lowerName.contains("test")) {
          Files.copy(dataFile.toPath(), outFile.toPath());          
        } else {
          Instances data = DataSource.read(dataFile.getCanonicalPath());
          data.setClassIndex(data.numAttributes() - 1);
          Constructor<? extends AbstractInstanceNoiseFilter> cons;
          AbstractInstanceNoiseFilter noiseFilter;
          if (keelNoiseFilter == null) {
            cons = filterClass.getConstructor(Instances.class);
            noiseFilter = cons.newInstance(data);
          } else {
            cons = filterClass.getConstructor(Instances.class, keelNoiseFilter.getClass());
            noiseFilter = cons.newInstance(data, keelNoiseFilter);
          }
          noiseFilter.apply();
          Instances filteredData = noiseFilter.safeExamples();
          DataSink.write(outFile.getCanonicalPath(), filteredData);
          avgFilteredPercent += 100.0 * ((double) noiseFilter.numFiltered() / (double) noiseFilter.numObjects());
          count ++;
        }
      }
      avgFilteredPercent /= count;
      fileWriter.write(String.format(" average percent of filtered instances  %4.2f \n", avgFilteredPercent));  
      fileWriter.flush();
    }
    fileWriter.write("\n");  
  }
  
  public void filterDataset(String path, 
          Class<? extends AbstractInstanceNoiseFilter> filterClass,
          Class<? extends KeelNoiseFilter> keelNoiseFilter, FileWriter fileWriter,
          String datasetName) throws IOException, Exception {
    File targetDir = new File((new File(path)).getCanonicalPath());
    File folder = new File(targetDir, datasetName);
    fileWriter.write("Filter Report \n");
    fileWriter.write("Path: " + folder + "\n\n");
    if (!outputDataDir.exists())
      outputDataDir.mkdirs();
    experimentDir.mkdir();
    if (keelNoiseFilter == null) {
      fileWriter.write("Filter scheme: " + filterClass.getName());
    }
    else {
      fileWriter.write("Filter scheme: " + filterClass.getName() + " + " + keelNoiseFilter.getName());
    }
    fileWriter.write("\n\n");  
    if (!folder.exists() || !folder.isDirectory()) {
      fileWriter.flush();
      throw new IOException("The supplied input path " + folder + " either does not exist or is not a valid directory");
    }
    fileWriter.write(String.format("%30s ", folder.getName())); 
    double avgFilteredPercent = 0;
    int count = 0;
    File outFolder = new File(experimentDir, folder.getName());
    outFolder.mkdir();
    File inCVFolder = new File(folder, "5-Folds-CV");
    File outCVFolder = new File(outFolder, inCVFolder.getName());
    outCVFolder.mkdir();
    File[] fileList = inCVFolder.listFiles();
    if (fileList == null) {
      throw new IOException("FileList null in folder: " + inCVFolder);
    }
    for (File dataFile : fileList) {
      File outFile = new File(outCVFolder, dataFile.getName());
      String lowerName = dataFile.getName().toLowerCase();
      if (lowerName.contains("test")) {
        Files.copy(dataFile.toPath(), outFile.toPath());          
      } else {
        Instances data = DataSource.read(dataFile.getCanonicalPath());
        data.setClassIndex(data.numAttributes() - 1);
        Constructor<? extends AbstractInstanceNoiseFilter> cons;
        AbstractInstanceNoiseFilter noiseFilter;
        if (keelNoiseFilter == null) {
          cons = filterClass.getConstructor(Instances.class);
          noiseFilter = cons.newInstance(data);
        } else {
          cons = filterClass.getConstructor(Instances.class, keelNoiseFilter.getClass());
          noiseFilter = cons.newInstance(data, keelNoiseFilter);
        }
        noiseFilter.apply();
        Instances filteredData = noiseFilter.safeExamples();
        DataSink.write(outFile.getCanonicalPath(), filteredData);
        avgFilteredPercent += 100.0 * ((double) noiseFilter.numFiltered() / (double) noiseFilter.numObjects());
        count ++;
      }
    }
    avgFilteredPercent /= count;
    fileWriter.write(String.format(" average percent of filtered instances  %4.2f \n", avgFilteredPercent));  
    fileWriter.flush();
    fileWriter.write("\n");  
  }

// *** METHODS FOR EVALUATING BENCHMARK DATA ***
  
  public void evaluateClassifiersOnBenchmarkData() {
    try {
      evalClassifiers(inputDataDir.getCanonicalPath());
    } catch (Exception ex) {
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void evalClassifiers(String path) throws IOException, Exception {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss.SSSS");
    String formatedDate = df.format(date);
    File report = new File(outputDir, "ClassificationReport-" + formatedDate + ".txt");
    FileWriter fileWriter = new FileWriter(report);
    fileWriter.write("Date " + formatedDate + "\n");
    evalClassifiers(path, fileWriter);
    fileWriter.close();
  }
  
  private void evaluateOnDataset(Classifier classifier, File folder, FileWriter fileWriter) throws IOException, Exception {
    fileWriter.write(String.format("*** %s *** \n\n", classifier.getClass().getName()));
    TrainArffFilter fileFilter = new TrainArffFilter();
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
    fileWriter.write("\n");
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
//      simpleMI.setOptions(new String[]{"-M", "2", "-W", "weka.classifiers.trees.J48", "--", "-U", "-M", "2"});
      evaluate(simpleMI, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador SimpleMI");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MITI miti = new MITI();
      evaluate(miti, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MITI");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MILR milr = new MILR();
      evaluate(milr, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MILR");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      CitationKNN citation = new CitationKNN();
      citation.setOptions(new String[]{"-R", "2", "-C", "2"});
      evaluate(citation, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador CitationKNN");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MIBoost boost = new MIBoost();
      boost.setOptions(new String[]{"-R", "10", "-B", "0", "-W", "weka.classifiers.bayes.NaiveBayes"});
      evaluate(boost, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MIBoost");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void evalClassifiersOnDataset(String path, FileWriter fileWriter,
    String datasetName) throws IOException {
    File inputDir = new File(path, datasetName);
    fileWriter.write("Classification Report \n");
    fileWriter.write("Path: " + inputDir + "\n\n");
    if (!inputDir.exists() || !inputDir.isDirectory()) {
      throw new IOException("The supplied input path either does not exist or not is a valid directory");
    }
    if (!outputDir.exists())
      outputDir.mkdirs();
    try {
      SimpleMI simpleMI = new SimpleMI();
      simpleMI.setOptions(new String[]{"-M", "2", "-W", "weka.classifiers.trees.J48"});
//      simpleMI.setOptions(new String[]{"-M", "2", "-W", "weka.classifiers.trees.J48", "--", "-U", "-M", "2"});
      evaluateOnDataset(simpleMI, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador SimpleMI");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MITI miti = new MITI();
      evaluateOnDataset(miti, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MITI");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MILR milr = new MILR();
      evaluateOnDataset(milr, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MILR");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      CitationKNN citation = new CitationKNN();
      citation.setOptions(new String[]{"-R", "2", "-C", "2"});
      evaluateOnDataset(citation, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador CitationKNN");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
    }
    try {
      MIBoost boost = new MIBoost();
      boost.setOptions(new String[]{"-R", "10", "-B", "0", "-W", "weka.classifiers.bayes.NaiveBayes"});
      evaluateOnDataset(boost, inputDir, fileWriter);
    } catch (Exception ex) {
      System.err.println("Problemas con el clasificador MIBoost");
      Logger.getLogger(InstanceLevelNoiseFilterTest.class.getName()).log(Level.SEVERE, null, ex);
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
  
  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    InstanceLevelNoiseFilterTest test = new InstanceLevelNoiseFilterTest();
//    test.classificationOnArtificialData(IterativeSimpleInstanceNoiseFilter.class, EFNoiseFilter.class);
//    BagLevelNoiseInsufflator gen = new BagLevelNoiseInsufflator(20);
//    gen.run();
//    test.evaluateClassifiersOnBenchmarkData();
//    test.testFilterOnBenchmarkData(IterativeNNInstanceNoiseFilter.class, IPFNoiseFilter.class);

    if (args.length == 0) {
      System.err.println("No arguments");
    } else {
      StringBuilder str = new StringBuilder();
      for (int i = 0; i < args.length; i++) {
        str.append(args[i]);
        if (i < args.length - 1)
          str.append(" ");
      }
      
//    test.testFilterOnBenchmarkData(IterativeNNInstanceNoiseMultiFilter.class, null);
      test.testFilterOnDataset(IterativeNNInstanceNoiseMultiFilter.class, null, str.toString());
//      test.testFilterOnDataset(IterativeNNInstanceNoiseFilter.class, IPFNoiseFilter.class, str.toString());

    }
}

}
