/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utils;

/**
 *
 * @author Danel
 */
public class NoiseFilterEvaluator {
  
  private int tp;
  private int tn;
  private int fp;
  private int fn;
  private NoiseFilter noiseFilter;
  private Boolean[] groundTruth;
  
  public NoiseFilterEvaluator() {
    // Macro-averaged version
  }
  
  public NoiseFilterEvaluator(NoiseFilter noiseFilter, Boolean[] groundTruth) throws Exception {
    part(noiseFilter, groundTruth);
    
    /**
     * Chequeo...
     */
    
//    System.out.println();
//    System.out.println("*** CHEQUEO ***");
//    System.out.println();
//    System.out.println("La longitud de GroundTruth tiene que coincidir con el tamaño del dataset:");
//    System.out.println("Longitud de GroundTruth: " + groundTruth.length);
//    System.out.println("Tamaño del dataset: " + noiseFilter.numObjects());
//    System.out.println();
//    
//    System.out.println("El número de objetos filtrados tiene que coincidir con el "
//            + "número de predicciones positivas");
//    System.out.println("Número de objetos filtrados: " + noiseFilter.numFiltered());
//    System.out.println("Número de predicciones positivas: " + (tp + fp));
//    System.out.println();
  }
  
  public final void part(NoiseFilter noiseFilter, Boolean[] groundTruth) throws Exception {
    if (noiseFilter == null) {
      throw new Exception("Null noise filter ");
    }
    if (groundTruth == null) {
      throw new Exception("Null ground truth ");
    }
    if (noiseFilter.numObjects() != groundTruth.length) {
      throw new Exception("The number of ground truth entries does not match the filter's decision number");
    }
    this.noiseFilter = noiseFilter;
    this.groundTruth = groundTruth;
    computeConfusionMatrix();
  }
  
  private void computeConfusionMatrix() {
    for (int i = 0; i < groundTruth.length; i++) {
      if (groundTruth[i]) {
        if (noiseFilter.decision(i)) {
          tp++;
        } else {
          fn++;
        }
      } else {
        if (noiseFilter.decision(i)) {
          fp++;
        } else {
          tn++;
        }
      }
    }
  }
  
  public int numInstances() {
    return tp + tn + fp + fn;
  }
  
  public double accuracy() {
    return (double)(tp + tn) / (double)(tp + tn + fp + fn);
  }
  
  public double precision() {
    return (double) tp / (double)(tp + fp);
  }
  
  public double recall() {
    return (double) tp / (double) (tp + fn);
  }
  
  public int truePositives() {
    return tp;
  }
  
  public int trueNegatives() {
    return tn;
  }
  
  public int falsePositives() {
    return fp;
  }
  
  public int falseNegatives() {
    return fn;
  }
  
  public double fscore() {
    double precision = precision();
    double recall = recall();
    if (precision == 0|| recall == 0) {
      return 0;
    } else {
      return (2 * precision * recall) / (precision + recall);
    }
  }
  
//  public String truePositiveNoiseInstances() {
//    StringBuilder output = new StringBuilder();
//    output.append(tp).append(" true positive instance(s): \n");
//    for (int i = 0; i < numInstances(); i++) {
//      if (groundTruth[i] && noiseFilter.decision(i)) {
//        output.append(noiseFilter.getInstance(i)).append("\n");
//      } 
//    }
//    return output.toString();
//  }
//  
//  public String falsePositiveNoiseInstances() {
//    StringBuilder output = new StringBuilder();
//    output.append(fp).append(" false positive instance(s): \n");
//    for (int i = 0; i < numInstances(); i++) {
//      if (!groundTruth[i] && noiseFilter.decision(i)) {
//        output.append(noiseFilter.getInstance(i)).append("\n");
//      }
//    }
//    return output.toString();
//  }
//  
//  public String trueNegativeNoiseInstances() {
//    StringBuilder output = new StringBuilder();  
//    output.append(tn).append(" true negative instance(s): \n");
//    for (int i = 0; i < numInstances(); i++) {
//      if (!groundTruth[i] && !noiseFilter.decision(i)) {
//        output.append(noiseFilter.getInstance(i)).append("\n");
//      }
//    }
//    return output.toString();
//  }
//  
//  public String falseNegativeNoiseInstances() {
//    StringBuilder output = new StringBuilder();
//    output.append(fn).append(" false negative instance(s): \n");
//    for (int i = 0; i < numInstances(); i++) {
//      if (groundTruth[i] && !noiseFilter.decision(i)) {
//        output.append(noiseFilter.getInstance(i)).append("\n");
//      } 
//    }
//    return output.toString();
//  }
  
}
