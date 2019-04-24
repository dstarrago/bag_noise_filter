/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataGenerator;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Logger;
import java.util.logging.Level;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;

/**
 *
 * @author Danel
 */
public class InstanceLevelNoiseInsufflator {
  
  private final String inPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1";
  private final String outPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/InsuflatedData";
  private int insuflationPercent;
  private final Random random;
  
  public InstanceLevelNoiseInsufflator(int insuflationPercent) {
    this.insuflationPercent = insuflationPercent;
    random = new Random();
  }
  
  public void setInsuflationPercent(int insuflationPercent) {
    this.insuflationPercent = insuflationPercent;
  }
  
  public void run() {
    File inDir = new File(inPath);
    File outDir = new File(outPath);
    if (!outDir.exists()) {
      outDir.mkdir();
    }
    File[] folderList = inDir.listFiles();
    for (File folder : folderList) {
      if (!folder.isDirectory()) {
        continue;
      }
      File outFolder = new File(outDir, folder.getName());
      outFolder.mkdir();
      File inCVFolder = new File(folder, "5-Folds-CV");
      File outCVFolder = new File(outFolder, inCVFolder.getName());
      outCVFolder.mkdir();
      File[] fileList = inCVFolder.listFiles();
      for (File dataFile : fileList) {
        try {
          File outFile = new File(outCVFolder, dataFile.getName());
          String lowerName = dataFile.getName().toLowerCase();
          if (lowerName.contains("test")) {
            Files.copy(dataFile.toPath(), outFile.toPath());          
          } else {
            Instances data = DataSource.read(dataFile.getCanonicalPath());
            data.setClassIndex(data.numAttributes() - 1);
            Instances newData = insuflate(data);
            DataSink.write(outFile.getCanonicalPath(), newData);
          }
        } catch (Exception ex) {
          Logger.getLogger(InstanceLevelNoiseInsufflator.class.getName()).log(Level.SEVERE, null, ex);
        }
      }
    }
  }
  
  private Instances insuflate(Instances data) {
    Instances newData = new Instances(data);
    ArrayList<Instance> classA = new ArrayList<>();
    ArrayList<Instance> classB = new ArrayList<>();
    for (int i = 0; i < newData.numInstances(); i++) {
      Instance bag = newData.get(i);
      if (bag.classValue() == 0) {
        classA.add(bag);
      } else {
        classB.add(bag);
      }
    }
    for (int i = 0; i < newData.numInstances(); i++) {
      Instance bag = newData.get(i);
      ArrayList<Instance> sampleClass;
      if (bag.classValue() == 0) {  // Bag in A
        sampleClass = classB;
      } else {                      // Bag in B
        sampleClass = classA;
      }
      Instances instances = bag.relationalValue(1);
      int numToInsuflate = (int) (instances.numInstances() * insuflationPercent / 100.0);
      if (numToInsuflate == 0 || sampleClass.isEmpty()) {
        continue;
      }
      for (int j = 0; j < numToInsuflate; j++) {
        instances.delete(instances.numInstances() - 1);
      }
      for (int j = 0; j < numToInsuflate; j++) {
        int bagIndex = random.nextInt(sampleClass.size());
        Instance sampleBag = sampleClass.get(bagIndex);
        int instIndex = random.nextInt(sampleBag.relationalValue(1).numInstances());
        Instance x = sampleBag.relationalValue(1).get(instIndex);
        instances.add(x);
      }
      instances.randomize(random);
    }
    return newData;
  }
  
}
