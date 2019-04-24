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
public class BagLevelNoiseInsufflator {
  
  private final String inPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/Bag Insuflated Noise 0%";
  private final String outPath = "C:/Users/Danel/Documents/Investigación/Proyectos/Noisy MIL/Experimentos/InsuflatedData";
  private int insuflationPercent;
  private final Random random;
  
  public BagLevelNoiseInsufflator(int insuflationPercent) {
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
          Logger.getLogger(BagLevelNoiseInsufflator.class.getName()).log(Level.SEVERE, null, ex);
        }
      }
    }
  }
  
  private Instances insuflate(Instances data) {
    Instances newData = new Instances(data);
    int numToInsuflate = (int) (newData.numInstances() * insuflationPercent / 100.0);
    ArrayList<Instance> exemplars = new ArrayList<>();
    exemplars.addAll(newData);
    for (int i = 0; i < numToInsuflate; i++) {
      int bagIndex = random.nextInt(exemplars.size());
      double newclassValue = 1 - exemplars.get(bagIndex).classValue();
      exemplars.get(bagIndex).setClassValue(newclassValue);
      exemplars.remove(bagIndex);
    }
    return newData;
  }
  
}
